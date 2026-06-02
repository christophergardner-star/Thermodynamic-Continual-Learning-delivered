"""One-shot script to write the Phase 2/3 experiment queue."""
import json, hashlib
from pathlib import Path
from datetime import datetime, timezone

queue_path = Path(r'E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state\experiment_queue.json')
PREG = Path(r'E:\TAR\Thermodynamic-Continual-Learning-delivered\tar_state\preregistrations')

def make_id(*parts):
    s = '|'.join(str(p) for p in parts)
    return hashlib.md5(s.encode()).hexdigest()[:16]

now = datetime.now(timezone.utc).isoformat()

experiments = [
    {
        'id': make_id('hp_selection', 'tcl_phd_rehab', 999),
        'name': 'HP Selection — Fair Joint (seed=999 validation)',
        'project_id': 'tcl_phd_rehabilitation',
        'hypothesis_name': 'fair_hyperparameter_selection',
        'dataset': 'split_cifar10',
        'method': 'multi_method',
        'seeds': [999],
        'config_overrides': {},
        'runner_key': 'hp_selection',
        'priority': 1,
        'estimated_runtime_h': 1.5,
        'backbone': 'resnet18',
        'epochs': 20,
        'description': 'Runs run_hyperparameter_selection.py — locks baseline HPs before confirmatory runs',
        'tags': ['phase2', 'prerequisite'],
        'status': 'pending',
        'stage': 'queued',
        'submitted_at': now,
    },
    {
        'id': make_id('hpc_replication', 'tcl_phd_rehab', 20),
        'name': 'HPC Replication n=20 fresh seeds SPRT (Phase 2.1)',
        'project_id': 'tcl_phd_rehabilitation',
        'hypothesis_name': 'hpc_replication_confirmatory',
        'dataset': 'split_cifar10',
        'method': 'hpc_tcl',
        'seeds': list(range(9, 29)),
        'config_overrides': {'tcl_penalty_lambda': 0.05},
        'runner_key': 'hpc_replication_phase2',
        'priority': 2,
        'estimated_runtime_h': 6.0,
        'backbone': 'resnet18',
        'epochs': 40,
        'description': 'Pre-registered SPRT replication of HPC finding (p=0.018). Confirmatory single-hypothesis.',
        'tags': ['phase2', 'hpc', 'pre_registered', 'sprt'],
        'status': 'pending',
        'stage': 'planned',
        'submitted_at': now,
        'pre_registration': str(PREG / 'hpc_replication.json'),
    },
    {
        'id': make_id('mechanistic_ablation_7c', 'tcl_phd_rehab', 3),
        'name': 'Mechanistic Ablation 7 conditions (Phase 3.1)',
        'project_id': 'tcl_phd_rehabilitation',
        'hypothesis_name': 'mechanistic_ablation_7condition',
        'dataset': 'split_cifar10',
        'method': 'tcl_ablation',
        'seeds': [42, 0, 1, 2, 3],
        'config_overrides': {'conditions': ['anchor_frozen_init', 'warmup_batches_60', 'ewc_best_lambda']},
        'runner_key': 'mechanistic_ablation_7c',
        'priority': 3,
        'estimated_runtime_h': 2.5,
        'backbone': 'resnet18',
        'epochs': 40,
        'description': 'Pre-registered 7-condition ablation. Runs 3 new; merges with Phase 11 (4 existing).',
        'tags': ['phase3', 'ablation', 'pre_registered', 'bonferroni_k6'],
        'status': 'pending',
        'stage': 'planned',
        'submitted_at': now,
        'pre_registration': str(PREG / 'mechanistic_ablation_7condition.json'),
    },
    {
        'id': make_id('phase16_cifar100_rerun', 'tcl_phd_rehab', 5),
        'name': 'Phase 16 CIFAR-100 Rerun 5 seeds 7 methods (Phase 2.3)',
        'project_id': 'tcl_phd_rehabilitation',
        'hypothesis_name': 'phase16_cifar100_confirmatory',
        'dataset': 'split_cifar100',
        'method': 'multi_method',
        'seeds': [42, 0, 1, 2, 3],
        'config_overrides': {},
        'runner_key': 'phase16_cifar100_rerun',
        'priority': 4,
        'estimated_runtime_h': 10.0,
        'backbone': 'resnet18',
        'epochs': 40,
        'description': 'Pre-registered 5-seed Phase 16 rerun with 7 methods. Upgrades EXPLORATION_GRADE if p<0.0125.',
        'tags': ['phase2', 'cifar100', 'pre_registered', 'scale_up'],
        'status': 'pending',
        'stage': 'planned',
        'submitted_at': now,
        'pre_registration': str(PREG / 'phase16_rerun.json'),
    },
    {
        'id': make_id('phase17_tinyimagenet_rerun', 'tcl_phd_rehab', 5),
        'name': 'Phase 17 TinyImageNet Rerun 5 seeds 7 methods (Phase 2.2)',
        'project_id': 'tcl_phd_rehabilitation',
        'hypothesis_name': 'phase17_tinyimagenet_confirmatory',
        'dataset': 'split_tinyimagenet',
        'method': 'multi_method',
        'seeds': [42, 0, 1, 2, 3],
        'config_overrides': {},
        'runner_key': 'phase17_tinyimagenet_rerun',
        'priority': 5,
        'estimated_runtime_h': 14.0,
        'backbone': 'resnet18',
        'epochs': 40,
        'description': 'Pre-registered 5-seed TinyImageNet rerun. Uses Arrow cache loader.',
        'tags': ['phase2', 'tinyimagenet', 'pre_registered', 'scale_up'],
        'status': 'pending',
        'stage': 'planned',
        'submitted_at': now,
        'pre_registration': str(PREG / 'phase17_rerun.json'),
    },
    {
        'id': make_id('hpc_lambda_momentum_abl', 'tcl_phd_rehab', 4),
        'name': 'HPC Lambda vs Momentum Ablation 4 conditions (Phase 3.4)',
        'project_id': 'tcl_phd_rehabilitation',
        'hypothesis_name': 'hpc_lambda_momentum_disentanglement',
        'dataset': 'split_cifar10',
        'method': 'hpc_ablation',
        'seeds': [42, 0, 1, 2, 3],
        'config_overrides': {},
        'runner_key': 'hpc_lambda_momentum_abl',
        'priority': 6,
        'estimated_runtime_h': 3.0,
        'backbone': 'resnet18',
        'epochs': 40,
        'description': 'Pre-registered 4-condition HPC disentanglement. Requires hpc_replication_phase2 first.',
        'tags': ['phase3', 'hpc', 'ablation', 'pre_registered'],
        'status': 'pending',
        'stage': 'planned',
        'submitted_at': now,
        'pre_registration': str(PREG / 'hpc_lambda_momentum_ablation.json'),
    },
]

queue = {
    'queue_name': 'phase2_phase3_phd_rehabilitation',
    'version': 2,
    'created_at': now,
    'note': 'TAR PhD Rehabilitation Plan Phase 2/3 pre-registered experiments. Executes in priority order via runner_key dispatch.',
    'experiments': experiments,
}

queue_path.write_text(json.dumps(queue, indent=2))
print(f'Queue written: {queue_path}')
print(f'Experiments queued: {len(experiments)}')
for e in experiments:
    print(f'  [{e["priority"]}] {e["name"]}  key={e["runner_key"]}')
