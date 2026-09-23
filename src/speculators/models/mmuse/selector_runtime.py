"""Stateless Selector runtime methods for MMuse host models."""

import torch


class MMuseSelectorMixin:
    """Provide Selector scoring, path selection, and teacher-row loss helpers.

    The host owns ``config``, ``candidate_selector``, ``block_size``, and
    ``draft_vocab_size``, and provides ``_draft_ids_to_verifier``. This mixin
    creates no modules, parameters, or other state.
    """

    def dflash2_select_candidates(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score one position's Top-K candidates for DFlash2 rollout.

        ``previous_token_ids`` are verifier-vocabulary IDs. Returned candidate
        IDs remain in the draft vocabulary, matching ``logits``.
        """
        if self.candidate_selector is None:
            raise RuntimeError("DFlash2 candidate selector is not enabled")
        if logits.shape[:-1] != hidden_states.shape[:-1]:
            raise ValueError("DFlash2 logits and hidden states must align")
        if previous_token_ids.shape != hidden_states.shape[:-1]:
            raise ValueError("DFlash2 previous-token IDs and hidden states must align")
        unary_logits, candidate_ids = torch.topk(
            logits,
            k=self.candidate_selector.top_k,
            dim=-1,
        )
        candidate_logits = self.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden_states,
            previous_token_ids,
        )
        return candidate_ids, candidate_logits

    def dflash2_sparse_logits(
        self,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
        previous_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the selector's realized sparse logits and Top-K draft IDs."""
        candidate_ids, candidate_logits = self.dflash2_select_candidates(
            logits,
            hidden_states,
            previous_token_ids,
        )
        sparse_logits = torch.full_like(logits, -torch.inf)
        sparse_logits.scatter_(-1, candidate_ids, candidate_logits)
        return sparse_logits, candidate_ids

    def _dflash2_select_topk_path(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_blocks: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select one block path from fixed DFlash Top-K candidates.

        Greedy mode reproduces the public DFlash2 walk. Global mode streams the
        block-local K-by-K edge lattice one position at a time and uses Viterbi
        backtracking to maximize the locally normalized joint path probability.
        Path selection is intentionally discrete; selector parameters are trained
        by the teacher-row loss in :meth:`_dflash2_block_outputs`.
        """
        if self.candidate_selector is None:
            raise RuntimeError("DFlash2 candidate selector is not enabled")
        if candidate_ids.shape != unary_logits.shape:
            raise ValueError("DFlash2 candidate IDs and unary logits must align")
        if candidate_ids.shape[:-1] != hidden_blocks.shape[:-1]:
            raise ValueError("DFlash2 candidates and hidden blocks must align")
        if anchor_token_ids.shape != (candidate_ids.shape[0],):
            raise ValueError("DFlash2 path selection requires one anchor per block")

        start_position = 0 if self.config.sample_from_anchor else 1
        selected_ids = candidate_ids[..., 0].detach().clone()
        realized_logits = unary_logits.detach().clone()
        active_candidates = candidate_ids[:, start_position:]
        if active_candidates.shape[1] == 0:
            return selected_ids, realized_logits

        active_unary = unary_logits[:, start_position:]
        active_hidden = hidden_blocks[:, start_position:]
        search_mode = self.config.dflash2_selector_search_mode
        with torch.no_grad():
            if search_mode == "greedy":
                selected_active, realized_active = self._dflash2_select_greedy_path(
                    active_candidates, active_unary, active_hidden, anchor_token_ids
                )
            elif search_mode == "global":
                selected_active, realized_active = self._dflash2_select_global_path(
                    active_candidates, active_unary, active_hidden, anchor_token_ids
                )
            else:
                raise ValueError(
                    f"Unsupported DFlash2 selector search mode: {search_mode!r}"
                )

        selected_ids[:, start_position:] = selected_active
        realized_logits[:, start_position:] = realized_active
        return selected_ids, realized_logits

    def _dflash2_select_greedy_path(
        self,
        active_candidates: torch.Tensor,
        active_unary: torch.Tensor,
        active_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Walk active positions; the dispatcher owns validation and no-grad."""
        previous_ids = anchor_token_ids.long()
        active_selected: list[torch.Tensor] = []
        active_rows: list[torch.Tensor] = []
        for position in range(active_candidates.shape[1]):
            row = self.candidate_selector(
                active_candidates[:, position],
                active_unary[:, position],
                active_hidden[:, position],
                previous_ids,
            )
            selected_indices = row.argmax(dim=-1, keepdim=True)
            selected_draft_ids = (
                active_candidates[:, position].gather(-1, selected_indices).squeeze(-1)
            )
            active_rows.append(row)
            active_selected.append(selected_draft_ids)
            previous_ids = self._draft_ids_to_verifier(selected_draft_ids)
        selected_active = torch.stack(active_selected, dim=1)
        realized_active = torch.stack(active_rows, dim=1)
        return selected_active, realized_active

    def _dflash2_select_global_path(
        self,
        active_candidates: torch.Tensor,
        active_unary: torch.Tensor,
        active_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stream normalized edges, backtrack, then rescore the realized path.

        The dispatcher owns validation and the no-grad search boundary.
        """
        # The first predecessor is the anchor. Later predecessor rows are
        # the verifier IDs corresponding to every prior Top-K candidate.
        first_row = self.candidate_selector(
            active_candidates[:, 0],
            active_unary[:, 0],
            active_hidden[:, 0],
            anchor_token_ids.long(),
        )
        # Selector rows define locally normalized conditional
        # distributions.  Viterbi must therefore add row log-probabilities,
        # not raw energies whose partition function varies by predecessor.
        best_scores = torch.log_softmax(first_row.float(), dim=-1)
        backpointers: list[torch.Tensor] = []
        for position in range(1, active_candidates.shape[1]):
            predecessor_ids = self._draft_ids_to_verifier(
                active_candidates[:, position - 1]
            )
            lattice = self.candidate_selector.score_lattice(
                active_candidates[:, position : position + 1],
                active_unary[:, position : position + 1],
                active_hidden[:, position : position + 1],
                predecessor_ids.unsqueeze(1),
            )[:, 0]
            edge_log_probs = torch.log_softmax(lattice.float(), dim=-1)
            path_scores = best_scores.unsqueeze(-1) + edge_log_probs
            best_scores, previous_indices = path_scores.max(dim=-2)
            backpointers.append(previous_indices)

        active_indices: list[torch.Tensor] = [best_scores.argmax(dim=-1)]
        for previous_indices in reversed(backpointers):
            active_indices.append(
                previous_indices.gather(-1, active_indices[-1].unsqueeze(-1)).squeeze(
                    -1
                )
            )
        active_indices.reverse()
        selected_indices = torch.stack(active_indices, dim=1)
        selected_active = active_candidates.gather(
            -1, selected_indices.unsqueeze(-1)
        ).squeeze(-1)

        active_rows = []
        previous_ids = anchor_token_ids.long()
        for position in range(active_candidates.shape[1]):
            active_rows.append(
                self.candidate_selector(
                    active_candidates[:, position],
                    active_unary[:, position],
                    active_hidden[:, position],
                    previous_ids,
                )
            )
            previous_ids = self._draft_ids_to_verifier(selected_active[:, position])
        realized_active = torch.stack(active_rows, dim=1)
        return selected_active, realized_active

    def dflash2_select_path(
        self,
        logits: torch.Tensor,
        hidden_blocks: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return Top-K IDs, their realized path rows, and selected draft IDs."""
        if self.candidate_selector is None:
            raise RuntimeError("DFlash2 candidate selector is not enabled")
        if logits.shape[:-1] != hidden_blocks.shape[:-1]:
            raise ValueError("DFlash2 logits and hidden blocks must align")
        unary_logits, candidate_ids = torch.topk(
            logits,
            k=self.candidate_selector.top_k,
            dim=-1,
        )
        selected_ids, realized_logits = self._dflash2_select_topk_path(
            candidate_ids,
            unary_logits,
            hidden_blocks,
            anchor_token_ids,
        )
        return candidate_ids, realized_logits, selected_ids

    def _dflash2_proposal_logits(
        self,
        candidate_ids: torch.Tensor,
        realized_logits: torch.Tensor,
        selected_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return candidate logits that reproduce the configured path proposal.

        The public greedy walk selects the local argmax of every realized row, so
        its original logits already describe the proposal. A Viterbi path can
        deliberately take a locally suboptimal edge to improve later positions;
        expose that deterministic whole-path decision as a one-hot distribution.
        """
        search_mode = getattr(self.config, "dflash2_selector_search_mode", "greedy")
        if search_mode == "greedy":
            return realized_logits
        if search_mode != "global":
            raise ValueError(
                f"Unsupported DFlash2 selector search mode: {search_mode!r}"
            )
        selected_mask = candidate_ids == selected_ids.unsqueeze(-1)
        if not selected_mask.any(dim=-1).all():
            raise RuntimeError("DFlash2 global path selected an ID outside its Top-K")
        return torch.where(
            selected_mask,
            torch.zeros_like(realized_logits),
            torch.full_like(realized_logits, -torch.inf),
        )

    def _dflash2_block_outputs(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        hidden_blocks: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        teacher_previous_token_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build selector rollout scores and its restricted-Top-K train loss.

        The public DFlash2 inference implementation exposes a K-by-K transition
        lattice but not its training objective. This repository directly scores
        the teacher/realized predecessor row and distils the verifier distribution
        restricted to the current Top-K. This avoids materializing an
        ``anchors x positions x K x K`` autograd graph. The existing
        full-vocabulary draft loss is untouched.
        """
        if self.candidate_selector is None:
            raise RuntimeError("DFlash2 candidate selector is not enabled")
        num_blocks, block_size, _ = hidden_blocks.shape
        if block_size != self.block_size:
            raise ValueError("DFlash2 hidden blocks use the wrong block size")
        expected_logits_shape = (1, num_blocks * block_size, self.draft_vocab_size)
        if logits.shape != expected_logits_shape or targets.shape != logits.shape:
            raise ValueError("DFlash2 logits/targets have an unexpected shape")
        if loss_mask.shape != logits.shape[:-1]:
            raise ValueError("DFlash2 loss mask must align with logits")
        if anchor_token_ids.shape != (num_blocks,):
            raise ValueError("DFlash2 requires one verifier anchor ID per block")
        if teacher_previous_token_ids.shape != (num_blocks, block_size):
            raise ValueError(
                "DFlash2 teacher predecessor IDs must align with hidden blocks"
            )
        logits_blocks = logits.view(num_blocks, block_size, -1)
        target_blocks = targets.view_as(logits_blocks)
        mask_blocks = loss_mask.view(num_blocks, block_size)
        unary_logits, candidate_ids = torch.topk(
            logits_blocks,
            k=self.candidate_selector.top_k,
            dim=-1,
        )
        start_position = 0 if self.config.sample_from_anchor else 1
        active_candidates = candidate_ids[:, start_position:]
        active_unary = unary_logits[:, start_position:]
        active_hidden = hidden_blocks[:, start_position:]
        active_targets = target_blocks[:, start_position:]
        active_mask = mask_blocks[:, start_position:]
        teacher_rows = self.candidate_selector(
            active_candidates,
            active_unary,
            active_hidden,
            teacher_previous_token_ids[:, start_position:],
        )
        selected_ids, realized_logits = self._dflash2_select_topk_path(
            candidate_ids,
            unary_logits,
            hidden_blocks,
            anchor_token_ids,
        )

        target_topk_logits = active_targets.gather(-1, active_candidates)
        target_topk_probs = torch.softmax(target_topk_logits.float(), dim=-1).detach()
        selector_loss_per_position = -(
            target_topk_probs * torch.log_softmax(teacher_rows.float(), dim=-1)
        ).sum(dim=-1)
        selector_mask = active_mask.to(selector_loss_per_position.dtype)
        selector_loss = (selector_loss_per_position * selector_mask).sum() / (
            selector_mask.sum().clamp_min(1.0)
        )
        full_teacher_rows = torch.cat(
            [unary_logits[:, :start_position], teacher_rows],
            dim=1,
        )
        return (
            candidate_ids,
            realized_logits,
            selector_loss,
            selected_ids,
            full_teacher_rows,
        )
