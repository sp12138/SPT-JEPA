"""Reject the known policy-target paths that violated the user's experiment rule.

Allowed inverse auxiliaries reconstruct a known executed transition/input. They
must be audited separately; naming a policy target 'current a_t' is no exception.
Read-only loading of historical models remains possible for forensic review.
"""
from pathlib import Path


def validate_training_supervision(config):
    model = config.get('model') or {}
    policy = config.get('model_class') in {
        'temporal_bridge', 'censored_action_bridge', 'correspondence_bridge',
    } and model.get('objective', 'flow') != 'metric'
    policy_data = config.get('dataset_kind') in {'native_inverse_pairs', 'local_inverse_mix'}
    declared_future_target = config.get('supervision_contract') == 'goal_conditioned_next_action_target'
    if policy or policy_data or declared_future_target:
        raise ValueError(
            'Forbidden next-to-execute action supervision: (current observation, '
            'desired/hindsight future goal) -> action is an action target even '
            'when named current a_t. Actions may condition forward prediction; '
            'an inverse auxiliary may reconstruct an already supplied/executed '
            'transition input, but this goal-conditioned policy path is disallowed.'
        )
    # An observation-only loss does not erase forbidden policy ancestry.
    references = {'output', 'init_checkpoint', 'initial_checkpoint',
                  'initial_model_checkpoint', 'initial_encoder_checkpoint',
                  'warm_start_checkpoint', 'teacher_checkpoint',
                  'distillation_checkpoint', 'feature_cache',
                  'proposal_bank', 'retrieval_bank'}
    for section in [config, model]:
        for key in references:
            value = section.get(key)
            if not isinstance(value, (str, Path)) or not value:
                continue
            path = Path(value).expanduser().resolve()
            for ancestor in [path, *path.parents]:
                if (ancestor/'DISQUALIFIED_ACTION_TARGET_SUPERVISION.json').exists():
                    raise ValueError(f'Forbidden disqualified policy lineage in {key}: {value}')
