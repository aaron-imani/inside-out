from utils import get_benchmarks_data, get_all_data
from concept_activation_analyzer import ConceptActivationAnalyzer
from pathlib import Path
import json
from tqdm import tqdm


cur_dir = Path(__file__).parent.resolve()
results_dir = cur_dir / "results"
results_dir.mkdir(exist_ok=True)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="Run Concept Activation Analysis")
    parser.add_argument("-m",'--models', type=str, nargs='+', default=['Qwen2.5-Coder', 'Qwen2.5', 'QwQ'], help='List of model names to analyze')
    parser.add_argument('-c', '--concepts', type=str, nargs='+', default=['comment', 'inline', 'multiline', 'javadoc'], help='List of concepts to analyze')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'individual'], help='Mode of analysis')
    args = parser.parse_args()


    # Load benchmarks data
    if args.mode == 'individual':
        benchmarks = get_benchmarks_data()
    else:
        benchmarks = {
            'combined': get_all_data()
        }

    for benchmark, data in benchmarks.items():
        print(f"Benchmark: {benchmark}, Comment Types: {list(data.keys())}")
        for comment_type, codes in data.items():
            print(f"  Comment Type: {comment_type}, Number of Codes: {len(codes)}")
        print('---' * 20)
 
    with open(cur_dir / 'prompts.json', 'r') as f:
        prompts = json.load(f)

    num_tasks = len(prompts)
    task_names = list(prompts.keys())

    models = args.models
    comment_types = args.concepts
    
    model_results = {
            comment_type: {benchmark: {task: [] for task in task_names} for benchmark in benchmarks.keys()}
            for comment_type in comment_types
    }

    for model_name in tqdm(models, desc="Processing models", unit="model"):
        analyzer = ConceptActivationAnalyzer(model_name=model_name, concept="comment")
        model_nickname = analyzer._model_name

        for j, (benchmark, data) in tqdm(enumerate(benchmarks.items()), desc="Processing benchmarks", unit="benchmark", leave=False):
            benchmark_comment_types = list(data.keys())
            col = j + 1

            for i, comment_type in tqdm(enumerate(comment_types), desc="Processing comment types", unit="comment type", leave=False):
                if comment_type not in benchmark_comment_types:
                    continue
                
                analyzer.set_classifier(comment_type)

                row = i + 1

                for task in prompts:
                    prompt_template = prompts[task]
                    task_prompts = [prompt_template.format(code=code) for code in data[comment_type]]
                    concept_activations = analyzer.analyze_concept_activation(
                        task_prompts
                    )

                    model_results[comment_type][benchmark][task] = concept_activations

        filepath = results_dir / f"{model_name}_concept_activations_{args.mode}.json" if args.mode == 'all' else results_dir / f"{model_name}_concept_activations.json"

        with open(filepath, 'w') as f:
            json.dump(model_results, f)