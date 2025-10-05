# AI Coding Agent Instructions for Sys-Scan Embedded Agent

## Project Overview
This codebase implements an end-to-end ML pipeline for training security analyst models that enhance the sys-scan-graph security scanning platform. The system transforms raw security scan data from Sys-Scan-Graph's C++ scanner into actionable intelligence through fine-tuned Llama-3 models running locally within a LangGraph orchestration framework, eliminating external LLM dependencies.

**Core Architecture:**
- **Synthetic Data Generation** (`synthetic_data/`): Producer agents generate realistic security scan findings (processes, network, kernel params, etc.) with correlation analysis and quality verification
- **ML Training Pipeline** (`ml_pipeline/`): Fine-tunes Llama-3-8B models using Lion optimizer, quantizes to 4-bit for local inference
- **AWS Integration**: SageMaker training, EC2 spot instances for compute, automated deployment workflows
- **Integration Target**: Models embed into sys-scan-graph's Intelligence Layer, replacing LLM API calls with local inference

## Sys-Scan-Graph Ecosystem Context

### Core Scanner (C++)
The foundation is a high-performance C++20 scanner that enumerates host security surfaces:
- **Process enumeration**: `/proc/*/status`, command lines, optional SHA256 hashing
- **Network analysis**: Socket states, listening services, connection patterns
- **Kernel security**: Parameters, modules, hardening settings
- **File permissions**: SUID/SGID binaries, world-writable files
- **Compliance checks**: PCI DSS, HIPAA, NIST CSF mappings
- **Output**: Deterministic JSON with stable schema versioning

### Intelligence Layer (Python)
Consumes Core Scanner JSON and adds advanced analytics:
- **15-stage pipeline**: Load → Correlate → Risk Score → LLM Summarize → Actions
- **LangGraph orchestration**: Cyclical reasoning workflows for complex analysis
- **Risk modeling**: Impact/Exposure/Anomaly/Confidence scoring with logistic calibration
- **Correlation engine**: Rule-based relationships, temporal analysis, ATT&CK mapping
- **Current bottleneck**: External LLM calls for summarization and remediation guidance

### ML Pipeline Role
This repository bridges the gap by:
1. **Synthetic data generation** mimicking Core Scanner output schema
2. **Model training** on security analysis tasks (correlation, summarization, actions)
3. **Quantization** to 4-bit GPTQ for embedded deployment
4. **Local inference integration** replacing external LLM dependencies

## Key Components & Data Flow
1. **Data Generation**: `SyntheticDataPipeline` orchestrates 8 producer types → correlations → verification → transformation
2. **Model Training**: Keras 3 multi-backend (TensorFlow/JAX/PyTorch) with Lion optimizer, distributed across GPUs
3. **Quantization**: GPTQ 4-bit compression for memory-efficient deployment
4. **Integration**: Local inference in LangGraph workflows, zero external API dependencies

## Critical Developer Workflows

### Synthetic Data Generation
```python
# Local development (recommended)
from synthetic_data_pipeline import run_synthetic_data_pipeline
result = run_synthetic_data_pipeline(
    output_path="dataset.json",
    conservative_parallel=True,  # Uses ~4 workers, safe for local
    producer_counts={"processes": 50, "network": 30, "kernel_params": 10}
)
```

### Model Training
```bash
# AWS SageMaker training
python sagemaker_training.py  # Launches ml.g5.4xlarge with Lion optimizer

# Local training
cd ml_pipeline && python train.py --epochs 3 --batch_size 12
```

### AWS Deployment
```bash
# Launch spot instance with GPU
./launch_g4dn.sh  # g4dn.xlarge with T4 GPU, spot pricing

# Check instance status
aws ec2 describe-instances --filters "Name=key-name,Values=sys-scan-agent_instance"
```

### Integration Testing
```bash
# Test model integration with Intelligence Layer
python -c "
from sys_scan_graph_agent import analyze_with_local_model
result = analyze_with_local_model('scan_output.json', 'quantized_model.gguf')
print(result['enriched_findings'])
"
```

## Project-Specific Patterns & Conventions

### Parallel Processing
- **Conservative Mode**: Always use `conservative_parallel=True` for local development (limits to 4 workers)
- **Resource Monitoring**: Pipeline automatically reduces workers if CPU/memory >80%
- **Small Datasets**: Sequential processing for datasets ≤2 items

### Data Schema Compliance
- All generated data must conform to `ground_truth_schema.json`
- Required fields: `enriched_findings`, `correlations`, `reductions`, `summaries`, `actions`
- Risk scores: 0-100 scale with subscores (impact, exposure, anomaly, confidence)
- Schema matches sys-scan-graph Intelligence Layer expectations

### Training Configuration
- **Lion Optimizer**: Use `keras.optimizers.Lion(learning_rate=2e-4, weight_decay=0.01)` for memory efficiency
- **Batch Size**: 12 per replica for distributed training (24 global)
- **Sequence Length**: 512 tokens with dynamic padding
- **Quantization**: GPTQ 4-bit with calibration dataset

### AWS Resource Management
- **Spot Instances**: Prefer spot pricing with max price limits (e.g., $0.20 for g4dn.xlarge)
- **Security Groups**: Use predefined groups (sg-0c3016e7d01243e5b)
- **User Data**: Installs Python, Jupyter, transformers stack automatically

## Integration Points

### External Dependencies
- **Transformers**: For model loading/tokenization
- **Keras 3**: Multi-backend training (TensorFlow primary)
- **Datasets**: Hugging Face for data loading
- **AWS CLI/SDK**: For SageMaker and EC2 management

### Cross-Component Communication
- Synthetic data → ML training via JSON datasets
- Training outputs → Quantized models for LangGraph integration
- AWS resources → Automated instance management via bash scripts
- **Target Integration**: Models replace LLM calls in sys-scan-graph Intelligence Layer

## Quality Assurance Patterns

### Verification Pipeline
```python
from advanced_verification_agent import AdvancedVerificationAgent
verifier = AdvancedVerificationAgent()
report = verifier.verify_dataset(findings, correlations)
# Check report["overall_status"] == "passed"
```

### Testing Approach
- Unit tests in `test_*.py` files for producers and pipeline
- Parallel performance tests for scaling validation
- Schema validation against `ground_truth_schema.json`
- Integration tests with sys-scan-graph Intelligence Layer

## Common Pitfalls to Avoid
- Never run parallel processing without `conservative_parallel=True` on local machines
- Always validate data against schema before training
- Use absolute paths for AWS CLI commands (e.g., `aws ec2 run-instances`)
- Quantization requires representative calibration data for accuracy
- Ensure model outputs match Intelligence Layer expectations for seamless integration

## Key Reference Files
- `synthetic_data/synthetic_data_pipeline.py`: Main pipeline orchestrator
- `ml_pipeline/train.py`: Distributed training with Lion optimizer
- `ground_truth_schema.json`: Data structure specification (matches sys-scan-graph)
- `launch_*.sh`: AWS deployment scripts
- `sagemaker_training.py`: Cloud training configuration
- **Integration Target**: `sys-scan-graph/agent/` (Intelligence Layer)</content>
<parameter name="filePath">.github/copilot-instructions.md