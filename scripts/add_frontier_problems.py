"""Add multi-domain frontier problems to frontier_problems.json."""
import json
import pathlib
from datetime import datetime, timezone

ws = pathlib.Path("E:/TAR/Thermodynamic-Continual-Learning-delivered")
fp_path = ws / "tar_state" / "frontier_problems.json"
data = json.loads(fp_path.read_text(encoding="utf-8"))
now = datetime.now(timezone.utc).isoformat()

new_problems = [
    {
        "id": "fp-nlp-continual-forgetting",
        "title": "Continual Text Classification Without Catastrophic Forgetting",
        "domain": "nlp_continual_learning",
        "description": (
            "NLP models fine-tuned sequentially on new text classification tasks rapidly forget "
            "previously learned categories. This prevents deploying adaptive language systems "
            "in production without repeated full retraining."
        ),
        "why_important": (
            "Text classification tasks evolve continuously in enterprise settings "
            "(new product categories, new intents, new topics). Forgetting is a hard barrier "
            "to deploying lifelong NLP systems in document routing, support automation, and content moderation."
        ),
        "industry_problem_title": "Continual Text Classification Without Catastrophic Forgetting",
        "global_problem_statement": (
            "Production NLP systems must learn new text categories over time without erasing "
            "knowledge of previously learned ones."
        ),
        "industry_contexts": ["enterprise NLP", "document routing", "content moderation", "support automation"],
        "well_known_problem": True,
        "solution_family": "EWC-NLP / Replay-NLP (under evaluation)",
        "solution_novelty_note": (
            "ewc_nlp and replay_nlp are TAR-internal adaptations of standard continual learning "
            "methods applied to TF-IDF + MLP text classifiers. Results are preliminary."
        ),
        "target_venues": ["NeurIPS", "EMNLP", "ACL", "NAACL"],
        "candidate_datasets": ["split_agnews", "split_dbpedia"],
        "candidate_backbones": ["tfidf_mlp", "distilbert"],
        "external_baselines": ["ewc_nlp", "replay_nlp", "sgd_baseline"],
        "research_guidance": (
            "Start with Split-AGNews (2 tasks, 4 classes) for fast signal. "
            "Compare sgd_baseline (naive sequential) vs ewc_nlp vs replay_nlp. "
            "Do not claim thermodynamic methods work on NLP until tested."
        ),
        "status": "active",
        "experiments_linked": [],
        "papers_linked": [],
        "breakthroughs_found": 0,
        "priority": 8,
        "created_at": now,
        "updated_at": now,
    },
    {
        "id": "fp-ood-generalization",
        "title": "Robustness to Distribution Shift in Deployed Vision Systems",
        "domain": "generalization",
        "description": (
            "Neural networks trained on clean data degrade significantly under natural distribution "
            "shifts (noise, blur, brightness changes). This gap between lab accuracy and real-world "
            "accuracy is a fundamental barrier to reliable deployment."
        ),
        "why_important": (
            "Autonomous vehicles, medical imaging, satellite analysis, and industrial inspection "
            "all encounter distribution shift in the field. A model 95% accurate in the lab "
            "may be 70% accurate under mild corruption."
        ),
        "industry_problem_title": "Robustness to Distribution Shift in Deployed Vision Systems",
        "global_problem_statement": (
            "Vision models must maintain reliable accuracy under the natural distribution shifts "
            "they will encounter after deployment, not just on held-out clean test sets."
        ),
        "industry_contexts": ["autonomous vehicles", "medical imaging", "satellite analysis", "industrial inspection"],
        "well_known_problem": True,
        "solution_family": "Data augmentation, robust training (under evaluation)",
        "solution_novelty_note": "TAR evaluates augmentation as a robustness intervention. No novel method claimed yet.",
        "target_venues": ["NeurIPS", "ICML", "ICLR", "CVPR"],
        "candidate_datasets": ["cifar10_corrupted"],
        "candidate_backbones": ["resnet18"],
        "external_baselines": ["standard", "augmentation"],
        "research_guidance": (
            "Train on clean CIFAR-10, evaluate under synthetic noise, blur, brightness corruptions. "
            "Compare standard SGD vs augmentation training. "
            "Report accuracy drop (clean minus corrupted) as the primary metric."
        ),
        "status": "active",
        "experiments_linked": [],
        "papers_linked": [],
        "breakthroughs_found": 0,
        "priority": 7,
        "created_at": now,
        "updated_at": now,
    },
    {
        "id": "fp-adversarial-safety",
        "title": "Neural Network Robustness Under Adversarial and Natural Perturbations",
        "domain": "ai_safety",
        "description": (
            "Deep neural networks are vulnerable to both carefully crafted adversarial inputs "
            "and natural corruptions. This fragility poses safety risks in high-stakes applications "
            "and undermines trust in AI-assisted decisions."
        ),
        "why_important": (
            "Safety-critical systems cannot tolerate input-dependent failures. "
            "Characterising and improving robustness is a direct AI safety requirement."
        ),
        "industry_problem_title": "Neural Network Robustness Under Adversarial and Natural Perturbations",
        "global_problem_statement": (
            "Neural networks in safety-critical settings must be reliably robust "
            "to both natural distribution shifts and adversarially crafted inputs."
        ),
        "industry_contexts": ["medical diagnostics", "autonomous driving", "security screening", "financial fraud detection"],
        "well_known_problem": True,
        "solution_family": "Augmentation-based training (under evaluation)",
        "solution_novelty_note": "TAR evaluates augmentation as a robustness intervention. No adversarial training implemented yet.",
        "target_venues": ["NeurIPS", "ICLR", "IEEE S&P", "USENIX Security"],
        "candidate_datasets": ["cifar10_corrupted"],
        "candidate_backbones": ["resnet18"],
        "external_baselines": ["standard", "augmentation"],
        "research_guidance": (
            "Measure accuracy drop under synthetic corruptions (noise, blur, brightness). "
            "Compare standard training vs augmentation. "
            "Do not claim adversarial robustness without running adversarial attack evaluation."
        ),
        "status": "active",
        "experiments_linked": [],
        "papers_linked": [],
        "breakthroughs_found": 0,
        "priority": 7,
        "created_at": now,
        "updated_at": now,
    },
    {
        "id": "fp-benchmark-eval-methodology",
        "title": "Reliable Evaluation Methodology for Continual Learning Claims",
        "domain": "evaluation_methodology",
        "description": (
            "Continual learning papers frequently make strong claims based on a single benchmark, "
            "a single seed, or metrics that do not reflect real deployment requirements. "
            "The field lacks a shared, rigorous evaluation protocol."
        ),
        "why_important": (
            "Unreliable evaluation inflates published numbers, slows real progress, "
            "and undermines trust in AI research. A trustworthy evaluation methodology "
            "is a precondition for honest comparison across methods."
        ),
        "industry_problem_title": "Reliable Evaluation Methodology for Continual Learning Claims",
        "global_problem_statement": (
            "The continual learning field needs a statistically rigorous evaluation protocol "
            "that separates real performance gains from measurement noise and benchmark overfitting."
        ),
        "industry_contexts": ["ML research", "applied AI", "AI governance", "academic publishing"],
        "well_known_problem": True,
        "solution_family": "Multi-seed statistical testing (TAR methodology)",
        "solution_novelty_note": (
            "TAR itself is an experiment in rigorous evaluation: multiple seeds, effect sizes, "
            "p-values, and trust tiers. This frontier problem asks whether TAR-style evaluation "
            "reveals different conclusions than single-seed published comparisons."
        ),
        "target_venues": ["NeurIPS", "ICML", "ICLR", "JMLR"],
        "candidate_datasets": ["split_cifar10"],
        "candidate_backbones": ["resnet18"],
        "external_baselines": ["ewc", "sgd_baseline"],
        "research_guidance": (
            "Run TCL vs EWC vs SGD on Split-CIFAR-10 with 5 seeds and report full statistical analysis. "
            "Compare conclusions to published single-seed numbers to quantify variance inflation. "
            "This is meta-research about evaluation, not a new learning algorithm."
        ),
        "status": "active",
        "experiments_linked": [],
        "papers_linked": [],
        "breakthroughs_found": 0,
        "priority": 6,
        "created_at": now,
        "updated_at": now,
    },
]

existing_ids = {p.get("id") for p in data.get("problems", [])}
added = 0
for p in new_problems:
    if p["id"] not in existing_ids:
        data["problems"].append(p)
        added += 1

data["saved_at"] = now
fp_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
print(f"Added {added} new frontier problems. Total: {len(data['problems'])}")
for p in data["problems"]:
    print(f"  [{p.get('domain', '?')}] {p['id']}: {p['title'][:55]}")
