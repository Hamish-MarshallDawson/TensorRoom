# RoomCraft AI: Local Agentic Interior Design Engine (CUDA)

A fully self-hosted, multi-modal AI engineering platform that parses real-world room photography into structurally accurate architectural redesigns, matches generated items against a local, disk-based product index, and outputs a localized procurement manifest. Built entirely for local NVIDIA hardware execution.

## 🚀 Key Architectural Pillars

* **Local Inference Isolation:** Zero network calls or external cloud dependencies. Designed to run completely offline via localized weights and direct CUDA processing.
* **Structural Line Consistency:** Separates foreground semantic masks from background room geometry to prevent structural warping (walls/floors) during creative style transfer.
* **Multi-Modal Local Matching:** Bridges purely generative pixels back to real-world store items using localized vector representations.

## 📦 Core Architecture Matrix

* **Inference Pipeline:** Localized Spatial Depth Model + Mask Generation Model + Diffusion-Based Inpainter
* **System Language Processing:** Localized Quantized Text LLM (8B parameter footprint scale)
* **Storage & Index Layer:** Local Dockerized Vector Instance + Local Disk Directory Cache
* **Task Distribution Layer:** Asynchronous Task Process Worker + In-Memory Message Cache

## 📂 Repository Structure

```text
├── configs/                 # Independent configuration mapping files
│   └── hardware.json        # CUDA device ID allocations, VRAM bounds, half-precision toggles
├── src/
│   ├── app/                 # Request entry point routing and status loops
│   ├── coordinator/         # Local LLM text parsing and json state checking
│   ├── vision_core/         # Pure object extraction, depth computing, and inpainting loops
│   └── local_index/         # Vector index initialization, retrieval, and inventory mappings
├── docker-compose.yml       # Spins up local cache brokers and localized db services
└── run_pipeline.py          # Main local processing gate