import os
import re
import requests
import html2text
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from fpdf import FPDF

# -------------------------------
# Configuration
# -------------------------------

# Maximum recursion depth to avoid infinite crawling.
MAX_DEPTH = 2  # Adjust as needed

# Directories to store output files.
MD_DIR = "markdown_files"
PDF_DIR = "pdf_files"
os.makedirs(MD_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# Seed URLs – these can be both Markdown and HTML pages.
seed_urls = [

    ("Troubleshooting", "https://docs.vllm.ai/en/latest/_sources/getting_started/troubleshooting.md"),
    ("FAQ", "https://docs.vllm.ai/en/latest/_sources/getting_started/faq.md"),
    ("Examples Index", "https://docs.vllm.ai/en/latest/_sources/getting_started/examples/examples_index.md"),
    ("Quickstart", "https://docs.vllm.ai/en/latest/_sources/getting_started/quickstart.md"),
    ("Installation", "https://docs.vllm.ai/en/latest/_sources/getting_started/installation/index.md"),
    ("Generative Models", "https://docs.vllm.ai/en/latest/_sources/models/generative_models.md"),
    ("Pooling Models", "https://docs.vllm.ai/en/latest/_sources/models/pooling_models.md"),
    ("List of Supported Models", "https://docs.vllm.ai/en/latest/_sources/models/supported_models.md"),
    ("Built-in Extensions", "https://docs.vllm.ai/en/latest/_sources/models/extensions/index.md"),
    ("Quantization", "https://docs.vllm.ai/en/latest/_sources/features/quantization/index.md"),
    ("LoRA Adapters", "https://docs.vllm.ai/en/latest/_sources/features/lora.md"),
    ("Structured Outputs", "https://docs.vllm.ai/en/latest/_sources/features/structured_outputs.md"),
    ("Reasoning Outputs", "https://docs.vllm.ai/en/latest/_sources/features/reasoning_outputs.md"),
    ("Tool Calling", "https://docs.vllm.ai/en/latest/_sources/features/tool_calling.md"),
    ("Compatibility Matrix", "https://docs.vllm.ai/en/latest/_sources/features/compatibility_matrix.md"),
    ("Speculative Decoding", "https://docs.vllm.ai/en/latest/_sources/features/spec_decode.md"),
    ("Disaggregated Prefilling", "https://docs.vllm.ai/en/latest/_sources/features/disagg_prefill.md"),
    ("Automatic Prefix Caching (features)", "https://docs.vllm.ai/en/latest/_sources/features/automatic_prefix_caching.md"),
    ("Production Metrics", "https://docs.vllm.ai/en/latest/_sources/serving/metrics.md"),
    ("Distributed Inference and Serving", "https://docs.vllm.ai/en/latest/_sources/serving/distributed_serving.md"),
    ("Multimodal Inputs", "https://docs.vllm.ai/en/latest/_sources/serving/multimodal_inputs.md"),
    ("OpenAI-Compatible Server", "https://docs.vllm.ai/en/latest/_sources/serving/openai_compatible_server.md"),
    ("Offline Inference (serving)", "https://docs.vllm.ai/en/latest/_sources/serving/offline_inference.md"),
    ("External Integrations (serving)", "https://docs.vllm.ai/en/latest/_sources/serving/integrations/index.md"),
    ("Usage Stats Collection", "https://docs.vllm.ai/en/latest/_sources/serving/usage_stats.md"),
    ("Environment Variables", "https://docs.vllm.ai/en/latest/_sources/serving/env_vars.md"),
    ("Engine Arguments", "https://docs.vllm.ai/en/latest/_sources/serving/engine_args.md"),
    ("External Integrations (deployment)", "https://docs.vllm.ai/en/latest/_sources/deployment/integrations/index.md"),
    ("Using Other Frameworks", "https://docs.vllm.ai/en/latest/_sources/deployment/frameworks/index.md"),
    ("Using Nginx", "https://docs.vllm.ai/en/latest/_sources/deployment/nginx.md"),
    ("Using Kubernetes", "https://docs.vllm.ai/en/latest/_sources/deployment/k8s.md"),
    ("Using Docker", "https://docs.vllm.ai/en/latest/_sources/deployment/docker.md"),
    ("Optimization and Tuning", "https://docs.vllm.ai/en/latest/_sources/performance/optimization.md"),
    ("Benchmark Suites", "https://docs.vllm.ai/en/latest/_sources/performance/benchmarks.md"),
    ("Architecture Overview", "https://docs.vllm.ai/en/latest/_sources/design/arch_overview.md"),
    ("Integration with HuggingFace", "https://docs.vllm.ai/en/latest/_sources/design/huggingface_integration.md"),
    ("vLLM’s Plugin System", "https://docs.vllm.ai/en/latest/_sources/design/plugin_system.md"),
    ("vLLM Paged Attention", "https://docs.vllm.ai/en/latest/_sources/design/kernel/paged_attention.md"),
    ("Multi-Modal Data Processing", "https://docs.vllm.ai/en/latest/_sources/design/mm_processing.md"),
    ("Automatic Prefix Caching (design)", "https://docs.vllm.ai/en/latest/_sources/design/automatic_prefix_caching.md"),
    ("Python Multiprocessing", "https://docs.vllm.ai/en/latest/_sources/design/multiprocessing.md"),
    ("V1 Design Documents - Automatic Prefix Caching", "https://docs.vllm.ai/en/latest/_sources/design/v1/prefix_caching.md"),
    ("Contributing to vLLM", "https://docs.vllm.ai/en/latest/_sources/contributing/overview.md"),
    ("Profiling vLLM", "https://docs.vllm.ai/en/latest/_sources/contributing/profiling/profiling_index.md"),
    ("Dockerfile", "https://docs.vllm.ai/en/latest/_sources/contributing/dockerfile/dockerfile.md"),
    ("Adding a New Model", "https://docs.vllm.ai/en/latest/_sources/contributing/model/index.md"),
    ("Implementing a Basic Model", "https://docs.vllm.ai/en/latest/_sources/contributing/model/basic.md"),
    ("Registering a Model to vLLM", "https://docs.vllm.ai/en/latest/_sources/contributing/model/registration.md"),
    ("Writing Unit Tests", "https://docs.vllm.ai/en/latest/_sources/contributing/model/tests.md"),
    ("Multi-Modal Support", "https://docs.vllm.ai/en/latest/_sources/contributing/model/multimodal.md"),
    ("Vulnerability Management", "https://docs.vllm.ai/en/latest/_sources/contributing/vulnerability_management.md"),
    ("Offline Inference (API)", "https://docs.vllm.ai/en/latest/_sources/api/offline_inference/index.md"),
    ("LLM Class", "https://docs.vllm.ai/en/latest/_sources/api/offline_inference/llm.md"),
    ("LLM Inputs", "https://docs.vllm.ai/en/latest/_sources/api/offline_inference/llm_inputs.md"),
    ("vLLM Engine", "https://docs.vllm.ai/en/latest/_sources/api/engine/index.md"),
    ("LLMEngine", "https://docs.vllm.ai/en/latest/_sources/api/engine/llm_engine.md"),
    ("AsyncLLMEngine", "https://docs.vllm.ai/en/latest/_sources/api/engine/async_llm_engine.md"),
    ("Inference Parameters", "https://docs.vllm.ai/en/latest/_sources/api/inference_params.md"),
    ("Multi-Modality", "https://docs.vllm.ai/en/latest/_sources/api/multimodal/index.md"),
    ("Input Definitions", "https://docs.vllm.ai/en/latest/_sources/api/multimodal/inputs.md"),
    ("Data Parsing", "https://docs.vllm.ai/en/latest/_sources/api/multimodal/parse.md"),
    ("Data Processing", "https://docs.vllm.ai/en/latest/_sources/api/multimodal/processing.md"),
    ("Memory Profiling", "https://docs.vllm.ai/en/latest/_sources/api/multimodal/profiling.md"),
    ("Registry", "https://docs.vllm.ai/en/latest/_sources/api/multimodal/registry.md"),
    ("Model Development", "https://docs.vllm.ai/en/latest/_sources/api/model/index.md"),
    ("Base Model Interfaces", "https://docs.vllm.ai/en/latest/_sources/api/model/interfaces_base.md"),
    ("Optional Interfaces", "https://docs.vllm.ai/en/latest/_sources/api/model/interfaces.md"),
    ("Model Adapters", "https://docs.vllm.ai/en/latest/_sources/api/model/adapters.md"),
    ("vLLM Blog", "https://docs.vllm.ai/en/latest/_sources/community/blog.md"),
    ("vLLM Meetups (community)", "https://docs.vllm.ai/en/latest/_sources/community/meetups.md"),
    ("Sponsors", "https://docs.vllm.ai/en/latest/_sources/community/sponsors.md"),
    ("Suggest Edit", "https://github.com/vllm-project/vllm/edit/main/docs/source/index.md"),
    ("Source File", "https://docs.vllm.ai/_sources/index.md"),
    ("Entrypoints", "https://docs.vllm.ai/en/latest/_sources/design/arch_overview.md#entrypoints"),
    ("LLM Engine (anchor)", "https://docs.vllm.ai/en/latest/_sources/design/arch_overview.md#llm-engine"),
    ("Worker", "https://docs.vllm.ai/en/latest/_sources/design/arch_overview.md#worker"),
    ("Model Runner", "https://docs.vllm.ai/en/latest/_sources/design/arch_overview.md#model-runner"),
    ("Model (anchor)", "https://docs.vllm.ai/en/latest/_sources/design/arch_overview.md#model"),
    ("Class Hierarchy", "https://docs.vllm.ai/en/latest/_sources/design/arch_overview.md#class-hierarchy"),
    ("How Plugins Work in vLLM", "https://docs.vllm.ai/en/latest/_sources/design/plugin_system.md#how-plugins-work-in-vllm"),
    ("How vLLM Discovers Plugins", "https://docs.vllm.ai/en/latest/_sources/design/plugin_system.md#how-vllm-discovers-plugins"),
    ("Types of Supported Plugins", "https://docs.vllm.ai/en/latest/_sources/design/plugin_system.md#types-of-supported-plugins"),
    ("Guidelines for Writing Plugins", "https://docs.vllm.ai/en/latest/_sources/design/plugin_system.md#guidelines-for-writing-plugins"),
    ("Compatibility Guarantee", "https://docs.vllm.ai/en/latest/_sources/design/plugin_system.md#compatibility-guarantee"),
    ("Inputs (paged attention)", "https://docs.vllm.ai/en/latest/_sources/design/kernel/paged_attention.md#inputs"),
    ("Concepts", "https://docs.vllm.ai/en/latest/_sources/design/kernel/paged_attention.md#concepts"),
    ("Query", "https://docs.vllm.ai/en/latest/_sources/design/kernel/paged_attention.md#query"),
    ("Key", "https://docs.vllm.ai/en/latest/_sources/design/kernel/paged_attention.md#key"),
    ("QK", "https://docs.vllm.ai/en/latest/_sources/design/kernel/paged_attention.md#qk"),
    ("Softmax", "https://docs.vllm.ai/en/latest/_sources/design/kernel/paged_attention.md#softmax"),
    ("Value", "https://docs.vllm.ai/en/latest/_sources/design/kernel/paged_attention.md#value"),
    ("LV", "https://docs.vllm.ai/en/latest/_sources/design/kernel/paged_attention.md#lv"),
    ("Output", "https://docs.vllm.ai/en/latest/_sources/design/kernel/paged_attention.md#output"),
    ("Prompt Replacement Detection", "https://docs.vllm.ai/en/latest/_sources/design/mm_processing.md#prompt-replacement-detection"),
    ("Tokenized Prompt Inputs", "https://docs.vllm.ai/en/latest/_sources/design/mm_processing.md#tokenized-prompt-inputs"),
    ("Processor Output Caching", "https://docs.vllm.ai/en/latest/_sources/design/mm_processing.md#processor-output-caching"),
    ("Generalized Caching Policy", "https://docs.vllm.ai/en/latest/_sources/design/automatic_prefix_caching.md#generalized-caching-policy"),
    ("Debugging", "https://docs.vllm.ai/en/latest/_sources/design/multiprocessing.md#debugging"),
    ("Introduction (multiprocessing)", "https://docs.vllm.ai/en/latest/_sources/design/multiprocessing.md#introduction"),
    ("Multiprocessing Methods", "https://docs.vllm.ai/en/latest/_sources/design/multiprocessing.md#multiprocessing-methods"),
    ("Compatibility with Dependencies", "https://docs.vllm.ai/en/latest/_sources/design/multiprocessing.md#compatibility-with-dependencies"),
    ("Current State (v0)", "https://docs.vllm.ai/en/latest/_sources/design/multiprocessing.md#current-state-v0"),
    ("Prior State in v1", "https://docs.vllm.ai/en/latest/_sources/design/multiprocessing.md#prior-state-in-v1"),
    ("Alternatives Considered", "https://docs.vllm.ai/en/latest/_sources/design/multiprocessing.md#alternatives-considered"),
    ("Future Work", "https://docs.vllm.ai/en/latest/_sources/design/multiprocessing.md#future-work"),
    ("Data Structure", "https://docs.vllm.ai/en/latest/_sources/design/v1/prefix_caching.md#data-structure"),
    ("Operations", "https://docs.vllm.ai/en/latest/_sources/design/v1/prefix_caching.md#operations"),
    ("Example", "https://docs.vllm.ai/en/latest/_sources/design/v1/prefix_caching.md#example"),
    ("License", "https://docs.vllm.ai/en/latest/_sources/contributing/overview.md#license"),
    ("Developing", "https://docs.vllm.ai/en/latest/_sources/contributing/overview.md#developing"),
    ("Testing", "https://docs.vllm.ai/en/latest/_sources/contributing/overview.md#testing"),
    ("Issues", "https://docs.vllm.ai/en/latest/_sources/contributing/overview.md#issues"),
    ("Pull Requests & Code Reviews", "https://docs.vllm.ai/en/latest/_sources/contributing/overview.md#pull-requests-code-reviews"),
    ("Thank You", "https://docs.vllm.ai/en/latest/_sources/contributing/overview.md#thank-you"),
    ("Example Commands and Usage", "https://docs.vllm.ai/en/latest/_sources/contributing/profiling/profiling_index.md#example-commands-and-usage"),
    ("Reporting Vulnerabilities", "https://docs.vllm.ai/en/latest/_sources/contributing/vulnerability_management.md#reporting-vulnerabilities"),
    ("Vulnerability Management Team", "https://docs.vllm.ai/en/latest/_sources/contributing/vulnerability_management.md#vulnerability-management-team"),
    ("Slack Discussion", "https://docs.vllm.ai/en/latest/_sources/contributing/vulnerability_management.md#slack-discussion"),
    ("Vulnerability Disclosure", "https://docs.vllm.ai/en/latest/_sources/contributing/vulnerability_management.md#vulnerability-disclosure"),
    ("Sampling Parameters", "https://docs.vllm.ai/en/latest/_sources/api/inference_params.md#sampling-parameters"),
    ("Pooling Parameters", "https://docs.vllm.ai/en/latest/_sources/api/inference_params.md#pooling-parameters"),
    ("Module Contents", "https://docs.vllm.ai/en/latest/_sources/api/multimodal/index.md#module-contents"),
    ("Submodules (multimodal)", "https://docs.vllm.ai/en/latest/_sources/api/multimodal/index.md#submodules"),
    ("Submodules (model)", "https://docs.vllm.ai/en/latest/_sources/api/model/index.md#submodules"),
    ("Index", "https://docs.vllm.ai/en/latest/_sources/genindex.md"),
    ("Python Module Index", "https://docs.vllm.ai/en/latest/_sources/py-modindex.md"),
]

# ---------- Additional Links (as plain URLs) ----------
additional_links = [
    "https://docs.vllm.ai/en/stable/",
    "http://docs.vllm.ai/getting_started/installation/index.html",
    "http://docs.vllm.ai/getting_started/installation/gpu/index.html",
    "http://docs.vllm.ai/getting_started/installation/cpu/index.html",
    "http://docs.vllm.ai/getting_started/installation/ai_accelerator/index.html",
    "http://docs.vllm.ai/getting_started/quickstart.html",
    "http://docs.vllm.ai/getting_started/examples/examples_index.html",
    "http://docs.vllm.ai/getting_started/examples/examples_offline_inference_index.html",
    "http://docs.vllm.ai/getting_started/examples/aqlm_example.html",
    "http://docs.vllm.ai/getting_started/examples/arctic.html",
    "http://docs.vllm.ai/getting_started/examples/audio_language.html",
    "http://docs.vllm.ai/getting_started/examples/basic.html",
    "http://docs.vllm.ai/getting_started/examples/basic_with_model_default_sampling.html",
    "http://docs.vllm.ai/getting_started/examples/chat.html",
    "http://docs.vllm.ai/getting_started/examples/chat_with_tools.html",
    "http://docs.vllm.ai/getting_started/examples/classification.html",
    "http://docs.vllm.ai/getting_started/examples/cli.html",
    "http://docs.vllm.ai/getting_started/examples/cpu_offload.html",
    "http://docs.vllm.ai/getting_started/examples/distributed.html",
    "http://docs.vllm.ai/getting_started/examples/embedding.html",
    "http://docs.vllm.ai/getting_started/examples/encoder_decoder.html",
    "http://docs.vllm.ai/getting_started/examples/florence2_inference.html",
    "http://docs.vllm.ai/getting_started/examples/gguf_inference.html",
    "http://docs.vllm.ai/getting_started/examples/llm_engine_example.html",
    "http://docs.vllm.ai/getting_started/examples/lora_with_quantization_inference.html",
    "http://docs.vllm.ai/getting_started/examples/mlpspeculator.html",
    "http://docs.vllm.ai/getting_started/examples/multilora_inference.html",
    "http://docs.vllm.ai/getting_started/examples/neuron.html",
    "http://docs.vllm.ai/getting_started/examples/neuron_int8_quantization.html",
    "http://docs.vllm.ai/getting_started/examples/openai.html",
    "http://docs.vllm.ai/getting_started/examples/pixtral.html",
    "http://docs.vllm.ai/getting_started/examples/prefix_caching.html",
    "http://docs.vllm.ai/getting_started/examples/profiling.html",
    "http://docs.vllm.ai/getting_started/examples/profiling_tpu.html",
    "http://docs.vllm.ai/getting_started/examples/ray_placement.html",
    "http://docs.vllm.ai/getting_started/examples/rlhf.html",
    "http://docs.vllm.ai/getting_started/examples/save_sharded_state.html",
    "http://docs.vllm.ai/getting_started/examples/scoring.html",
    "http://docs.vllm.ai/getting_started/examples/simple_profiling.html",
    "http://docs.vllm.ai/getting_started/examples/structured_outputs.html",
    "http://docs.vllm.ai/getting_started/examples/torchrun_example.html",
    "http://docs.vllm.ai/getting_started/examples/tpu.html",
    "http://docs.vllm.ai/getting_started/examples/vision_language.html",
    "http://docs.vllm.ai/getting_started/examples/vision_language_embedding.html",
    "http://docs.vllm.ai/getting_started/examples/vision_language_multi_image.html",
    "http://docs.vllm.ai/getting_started/examples/whisper.html",
    "http://docs.vllm.ai/getting_started/examples/examples_online_serving_index.html",
    "http://docs.vllm.ai/getting_started/examples/api_client.html",
    "http://docs.vllm.ai/getting_started/examples/chart-helm.html",
    "http://docs.vllm.ai/getting_started/examples/cohere_rerank_client.html",
    "http://docs.vllm.ai/getting_started/examples/disaggregated_prefill.html",
    "http://docs.vllm.ai/getting_started/examples/gradio_openai_chatbot_webserver.html",
    "http://docs.vllm.ai/getting_started/examples/gradio_webserver.html",
    "http://docs.vllm.ai/getting_started/examples/jinaai_rerank_client.html",
    "http://docs.vllm.ai/getting_started/examples/openai_chat_completion_client.html",
    "http://docs.vllm.ai/getting_started/examples/openai_chat_completion_client_for_multimodal.html",
    "http://docs.vllm.ai/getting_started/examples/openai_chat_completion_client_with_tools.html",
    "http://docs.vllm.ai/getting_started/examples/openai_chat_completion_structured_outputs.html",
    "http://docs.vllm.ai/getting_started/examples/openai_chat_completion_with_reasoning.html",
    "http://docs.vllm.ai/getting_started/examples/openai_chat_completion_with_reasoning_streaming.html",
    "http://docs.vllm.ai/getting_started/examples/openai_chat_embedding_client_for_multimodal.html",
    "http://docs.vllm.ai/getting_started/examples/openai_completion_client.html",
    "http://docs.vllm.ai/getting_started/examples/openai_cross_encoder_score.html",
    "http://docs.vllm.ai/getting_started/examples/openai_embedding_client.html",
    "http://docs.vllm.ai/getting_started/examples/openai_pooling_client.html",
    "http://docs.vllm.ai/getting_started/examples/opentelemetry.html",
    "http://docs.vllm.ai/getting_started/examples/prometheus_grafana.html",
    "http://docs.vllm.ai/getting_started/examples/run_cluster.html",
    "http://docs.vllm.ai/getting_started/examples/sagemaker-entrypoint.html",
    "http://docs.vllm.ai/getting_started/examples/examples_other_index.html",
    "http://docs.vllm.ai/getting_started/examples/logging_configuration.html",
    "http://docs.vllm.ai/getting_started/examples/tensorize_vllm_model.html",
    "http://docs.vllm.ai/getting_started/troubleshooting.html",
    "http://docs.vllm.ai/getting_started/faq.html",
    "http://docs.vllm.ai/models/generative_models.html",
    "http://docs.vllm.ai/models/pooling_models.html",
    "http://docs.vllm.ai/models/supported_models.html",
    "http://docs.vllm.ai/models/extensions/index.html",
    "http://docs.vllm.ai/models/extensions/runai_model_streamer.html",
    "http://docs.vllm.ai/models/extensions/tensorizer.html",
    "http://docs.vllm.ai/features/quantization/index.html",
    "http://docs.vllm.ai/features/quantization/supported_hardware.html",
    "http://docs.vllm.ai/features/quantization/auto_awq.html",
    "http://docs.vllm.ai/features/quantization/bnb.html",
    "http://docs.vllm.ai/features/quantization/gguf.html",
    "http://docs.vllm.ai/features/quantization/int4.html",
    "http://docs.vllm.ai/features/quantization/int8.html",
    "http://docs.vllm.ai/features/quantization/fp8.html",
    "http://docs.vllm.ai/features/quantization/quantized_kvcache.html",
    "http://docs.vllm.ai/features/lora.html",
    "http://docs.vllm.ai/features/tool_calling.html",
    "http://docs.vllm.ai/features/reasoning_outputs.html",
    "http://docs.vllm.ai/features/structured_outputs.html",
    "http://docs.vllm.ai/features/automatic_prefix_caching.html",
    "http://docs.vllm.ai/features/disagg_prefill.html",
    "http://docs.vllm.ai/features/spec_decode.html",
    "http://docs.vllm.ai/features/compatibility_matrix.html",
    "http://docs.vllm.ai/serving/offline_inference.html",
    "http://docs.vllm.ai/serving/openai_compatible_server.html",
    "http://docs.vllm.ai/serving/multimodal_inputs.html",
    "http://docs.vllm.ai/serving/distributed_serving.html",
    "http://docs.vllm.ai/serving/metrics.html",
    "http://docs.vllm.ai/serving/engine_args.html",
    "http://docs.vllm.ai/serving/env_vars.html",
    "http://docs.vllm.ai/serving/usage_stats.html",
    "http://docs.vllm.ai/serving/integrations/index.html",
    "http://docs.vllm.ai/serving/integrations/langchain.html",
    "http://docs.vllm.ai/serving/integrations/llamaindex.html",
    "http://docs.vllm.ai/deployment/docker.html",
    "http://docs.vllm.ai/deployment/k8s.html",
    "http://docs.vllm.ai/deployment/nginx.html",
    "http://docs.vllm.ai/deployment/frameworks/index.html",
    "http://docs.vllm.ai/deployment/frameworks/bentoml.html",
    "http://docs.vllm.ai/deployment/frameworks/cerebrium.html",
    "http://docs.vllm.ai/deployment/frameworks/dstack.html",
    "http://docs.vllm.ai/deployment/frameworks/helm.html",
    "http://docs.vllm.ai/deployment/frameworks/lws.html",
    "http://docs.vllm.ai/deployment/frameworks/modal.html",
    "http://docs.vllm.ai/deployment/frameworks/skypilot.html",
    "http://docs.vllm.ai/deployment/frameworks/triton.html",
    "http://docs.vllm.ai/deployment/integrations/index.html",
    "http://docs.vllm.ai/deployment/integrations/kserve.html",
    "http://docs.vllm.ai/deployment/integrations/kubeai.html",
    "http://docs.vllm.ai/deployment/integrations/llamastack.html",
    "http://docs.vllm.ai/performance/optimization.html",
    "http://docs.vllm.ai/performance/benchmarks.html",
    "http://docs.vllm.ai/design/arch_overview.html",
    "http://docs.vllm.ai/design/huggingface_integration.html",
    "http://docs.vllm.ai/design/plugin_system.html",
    "http://docs.vllm.ai/design/kernel/paged_attention.html",
    "http://docs.vllm.ai/design/mm_processing.html",
    "http://docs.vllm.ai/design/automatic_prefix_caching.html",
    "http://docs.vllm.ai/design/multiprocessing.html",
    "http://docs.vllm.ai/design/v1/prefix_caching.html",
    "http://docs.vllm.ai/contributing/overview.html",
    "http://docs.vllm.ai/contributing/profiling/profiling_index.html",
    "http://docs.vllm.ai/contributing/dockerfile/dockerfile.html",
    "http://docs.vllm.ai/contributing/model/index.html",
    "http://docs.vllm.ai/contributing/model/basic.html",
    "http://docs.vllm.ai/contributing/model/registration.html",
    "http://docs.vllm.ai/contributing/model/tests.html",
    "http://docs.vllm.ai/contributing/model/multimodal.html",
    "http://docs.vllm.ai/contributing/vulnerability_management.html",
    "http://docs.vllm.ai/api/offline_inference/index.html",
    "http://docs.vllm.ai/api/offline_inference/llm.html",
    "http://docs.vllm.ai/api/offline_inference/llm_inputs.html",
    "http://docs.vllm.ai/api/engine/index.html",
    "http://docs.vllm.ai/api/engine/llm_engine.html",
    "http://docs.vllm.ai/api/engine/async_llm_engine.html",
    "http://docs.vllm.ai/api/inference_params.html",
    "http://docs.vllm.ai/api/multimodal/index.html",
    "http://docs.vllm.ai/api/multimodal/inputs.html",
    "http://docs.vllm.ai/api/multimodal/parse.html",
    "http://docs.vllm.ai/api/multimodal/processing.html",
    "http://docs.vllm.ai/api/multimodal/profiling.html",
    "http://docs.vllm.ai/api/multimodal/registry.html",
    "http://docs.vllm.ai/api/model/index.html",
    "http://docs.vllm.ai/api/model/interfaces_base.html",
    "http://docs.vllm.ai/api/model/interfaces.html",
    "http://docs.vllm.ai/api/model/adapters.html",
    "http://docs.vllm.ai/community/blog.html",
    "http://docs.vllm.ai/community/meetups.html",
    "http://docs.vllm.ai/community/sponsors.html",
    "https://github.com/vllm-project/vllm",
    "https://github.com/vllm-project/vllm/edit/main/docs/source/index.md",
    "http://docs.vllm.ai/_sources/index.md",
    "https://github.com/vllm-project/vllm/subscription",
    "https://github.com/vllm-project/vllm/fork",
    "https://sky.cs.berkeley.edu",
    "https://blog.vllm.ai/2023/06/20/vllm.html",
    "https://arxiv.org/abs/2210.17323",
    "https://arxiv.org/abs/2306.00978",
    "https://vllm.ai",
    "https://arxiv.org/abs/2309.06180",
    "https://www.anyscale.com/blog/continuous-batching-llm-inference",
    "http://docs.vllm.ai/community/meetups.html#meetups",
    "http://docs.vllm.ai/design/arch_overview.html#entrypoints",
    "http://docs.vllm.ai/design/arch_overview.html#llm-engine",
    "http://docs.vllm.ai/design/arch_overview.html#worker",
    "http://docs.vllm.ai/design/arch_overview.html#model-runner",
    "http://docs.vllm.ai/design/arch_overview.html#model",
    "http://docs.vllm.ai/design/arch_overview.html#class-hierarchy",
    "http://docs.vllm.ai/design/plugin_system.html#how-plugins-work-in-vllm",
    "http://docs.vllm.ai/design/plugin_system.html#how-vllm-discovers-plugins",
    "http://docs.vllm.ai/design/plugin_system.html#types-of-supported-plugins",
    "http://docs.vllm.ai/design/plugin_system.html#guidelines-for-writing-plugins",
    "http://docs.vllm.ai/design/plugin_system.html#compatibility-guarantee",
    "http://docs.vllm.ai/design/kernel/paged_attention.html#inputs",
    "http://docs.vllm.ai/design/kernel/paged_attention.html#concepts",
    "http://docs.vllm.ai/design/kernel/paged_attention.html#query",
    "http://docs.vllm.ai/design/kernel/paged_attention.html#key",
    "http://docs.vllm.ai/design/kernel/paged_attention.html#qk",
    "http://docs.vllm.ai/design/kernel/paged_attention.html#softmax",
    "http://docs.vllm.ai/design/kernel/paged_attention.html#value",
    "http://docs.vllm.ai/design/kernel/paged_attention.html#lv",
    "http://docs.vllm.ai/design/kernel/paged_attention.html#output",
    "http://docs.vllm.ai/design/mm_processing.html#prompt-replacement-detection",
    "http://docs.vllm.ai/design/mm_processing.html#tokenized-prompt-inputs",
    "http://docs.vllm.ai/design/mm_processing.html#processor-output-caching",
    "http://docs.vllm.ai/design/automatic_prefix_caching.html#generalized-caching-policy",
    "http://docs.vllm.ai/design/multiprocessing.html#debugging",
    "http://docs.vllm.ai/design/multiprocessing.html#introduction",
    "http://docs.vllm.ai/design/multiprocessing.html#multiprocessing-methods",
    "http://docs.vllm.ai/design/multiprocessing.html#compatibility-with-dependencies",
    "http://docs.vllm.ai/design/multiprocessing.html#current-state-v0",
    "http://docs.vllm.ai/design/multiprocessing.html#prior-state-in-v1",
    "http://docs.vllm.ai/design/multiprocessing.html#alternatives-considered",
    "http://docs.vllm.ai/design/multiprocessing.html#future-work",
    "http://docs.vllm.ai/design/v1/prefix_caching.html#data-structure",
    "http://docs.vllm.ai/design/v1/prefix_caching.html#operations",
    "http://docs.vllm.ai/design/v1/prefix_caching.html#example",
    "http://docs.vllm.ai/contributing/overview.html#license",
    "http://docs.vllm.ai/contributing/overview.html#developing",
    "http://docs.vllm.ai/contributing/overview.html#testing",
    "http://docs.vllm.ai/contributing/overview.html#issues",
    "http://docs.vllm.ai/contributing/overview.html#pull-requests-code-reviews",
    "http://docs.vllm.ai/contributing/overview.html#thank-you",
    "http://docs.vllm.ai/contributing/profiling/profiling_index.html#example-commands-and-usage",
    "http://docs.vllm.ai/contributing/vulnerability_management.html#reporting-vulnerabilities",
    "http://docs.vllm.ai/contributing/vulnerability_management.html#vulnerability-management-team",
    "http://docs.vllm.ai/contributing/vulnerability_management.html#slack-discussion",
    "http://docs.vllm.ai/contributing/vulnerability_management.html#vulnerability-disclosure",
    "http://docs.vllm.ai/api/inference_params.html#sampling-parameters",
    "http://docs.vllm.ai/api/inference_params.html#pooling-parameters",
    "http://docs.vllm.ai/api/multimodal/index.html#module-contents",
    "http://docs.vllm.ai/api/multimodal/index.html#submodules",
    "http://docs.vllm.ai/api/model/index.html#submodules",
    "http://docs.vllm.ai/genindex.html",
    "http://docs.vllm.ai/py-modindex.html"

]

# Normalize the seed list so that each item is a (title, url) tuple.
def normalize_seed(seed):
    if isinstance(seed, tuple):
        return seed
    else:
        # If only URL is provided, generate a title from the URL.
        return (title_from_url(seed), seed)

seed_urls = [normalize_seed(item) for item in seed_urls]

# -------------------------------
# Global visited set to avoid processing the same URL twice.
# -------------------------------
visited = set()

# -------------------------------
# Helper Functions
# -------------------------------

def safe_filename(name):
    """Return a safe filename by replacing non-alphanumeric characters with underscores."""
    return re.sub(r'\W+', '_', name)

def title_from_url(url):
    """Generate a simple title from a URL."""
    parsed = urlparse(url)
    # Remove query parameters and fragments.
    path = parsed.path
    title = os.path.basename(path) or parsed.netloc
    return title if title else "index"

def download_content(url):
    """
    Download the content from a URL.
    Returns a tuple (content, content_type) where content_type is either 'markdown' or 'html'.
    """
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {url}: Status code {response.status_code}")
            return None, None

        content = response.text
        # If URL ends with .md or the response Content-Type header contains "text/markdown"
        if url.lower().endswith('.md') or "text/markdown" in response.headers.get("Content-Type", ""):
            return content, "markdown"
        else:
            return content, "html"
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None, None

def convert_to_markdown(content, content_type):
    """
    Convert content to Markdown.
    If already markdown, return as is.
    If HTML, use html2text to convert.
    """
    if content_type == "markdown":
        return content
    else:
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        markdown_text = converter.handle(content)
        return markdown_text

def save_markdown(url, markdown_text):
    """
    Save the markdown text to a file.
    Returns the file path.
    """
    filename = safe_filename(url) + ".md"
    filepath = os.path.join(MD_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    return filepath

def markdown_to_pdf(markdown_text, pdf_filepath):
    """
    Convert markdown (or plain text) to PDF using FPDF.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in markdown_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    pdf.output(pdf_filepath)

def extract_links(url, content, content_type):
    """
    Extract absolute links from the page content.
    For HTML content, use BeautifulSoup.
    For Markdown content, use a regex to find Markdown-style links.
    Returns a set of absolute URLs.
    """
    links = set()
    if content_type == "html":
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = tag['href'].strip()
            # Resolve relative URLs.
            absolute = urljoin(url, href)
            # Only consider http(s) links.
            if absolute.startswith("http"):
                links.add(absolute)
    else:
        # Use a regex to extract Markdown links of the form [text](http://...)
        matches = re.findall(r'\[.*?\]\((https?://.*?)\)', content)
        for link in matches:
            links.add(link.strip())
    return links

# -------------------------------
# Recursive Processing Function
# -------------------------------

def process_url(url, depth=0):
    """
    Process a given URL:
      - Download the content.
      - Convert it to Markdown.
      - Save as a .md file.
      - Convert the Markdown to a PDF.
      - Extract links from the content.
      - Recursively process new links (if depth < MAX_DEPTH).
    """
    if url in visited:
        return
    visited.add(url)
    print(f"{'  '*depth}Processing: {url}")

    content, content_type = download_content(url)
    if content is None:
        print(f"{'  '*depth}Skipping {url} due to download error.")
        return

    # Convert content to Markdown (if needed)
    markdown_text = convert_to_markdown(content, content_type)

    # Save Markdown file
    md_filepath = save_markdown(url, markdown_text)
    print(f"{'  '*depth}Saved Markdown: {md_filepath}")

    # Convert Markdown to PDF
    pdf_filename = safe_filename(url) + ".pdf"
    pdf_filepath = os.path.join(PDF_DIR, pdf_filename)
    try:
        markdown_to_pdf(markdown_text, pdf_filepath)
        print(f"{'  '*depth}Saved PDF: {pdf_filepath}")
    except Exception as e:
        print(f"{'  '*depth}Error converting {url} to PDF: {e}")

    # If maximum recursion depth reached, do not follow links.
    if depth >= MAX_DEPTH:
        return

    # Extract new links and process them recursively.
    new_links = extract_links(url, content, content_type)
    print(f"{'  '*depth}Found {len(new_links)} new link(s).")
    for link in new_links:
        # (Optional) You could restrict crawling to certain domains here.
        process_url(link, depth + 1)

# -------------------------------
# Main: Process All Seed URLs
# -------------------------------
def main():
    for title, url in seed_urls:
        process_url(url, depth=0)

if __name__ == "__main__":
    main()
