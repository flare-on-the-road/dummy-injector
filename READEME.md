# 모드 1 — 전체 파이프라인 (기본)
python main.py fire.jpg --cctv goduck_tunnel --cctv-name "[세종] 고덕터널" --location 고덕터널

# 모드 2 — VLM 단독
python main.py bbox_fire.jpg --cctv goduck_tunnel --cctv-name "[세종] 고덕터널" --location 고덕터널 --mode vlm