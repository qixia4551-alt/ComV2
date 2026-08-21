============================================================
   ComfyUI Console - Usage Guide
============================================================

[QUICK START]
1. Make sure ComfyUI / Comfy Desktop is running first
2. Double-click "start.bat" to launch the console server
3. Open your browser and go to: http://127.0.0.1:8501
4. Default password: admin
5. Go to Settings to verify the ComfyUI API address

[FILES INCLUDED]
- start.bat     : Launch script (double-click to start)
- server.py     : Python backend server
- index.html    : Frontend web interface
- static/       : Static assets folder (optional)
- README.txt    : This file

[REQUIREMENTS]
- Python 3.6 or higher (no extra packages needed)
- ComfyUI running (any version)
- ComfyUI must have CORS enabled (--enable-cors-header *)
  Note: Comfy Desktop has CORS enabled by default

[COMFY DESKTOP USERS]
- Auto-detected: The console automatically finds Comfy Desktop
  model paths and settings
- Default API port: 8189 (not 8188!)
- Models stored at: D:\Comfy-Desktop\ComfyUI-Shared\models
- If API shows "Offline", check Settings > ComfyUI Connection
  and make sure the port is 8189

[HOW TO RUN COMFYUI WITH CORS (manual install)]
Edit your ComfyUI start script and add:
  --enable-cors-header *
For mobile access, also add:
  --listen 0.0.0.0

Example:
  python main.py --enable-cors-header * --listen 0.0.0.0

[FEATURES]
- Text to Image generation with full parameter control
- Gallery with history of generated images
- JSON Workflow editor for advanced users
- Settings panel (theme, password, VRAM monitor)
- Tag system (5 categories, click to insert English tags)
- Template system (save/apply prompt templates)
- Responsive design (desktop + mobile)
- Real-time progress via WebSocket
- Confetti effects on successful generation
- Local model auto-detection (checkpoints, VAE, LoRA)
- Manual model management (add models by name)
- Comfy Desktop auto-detection

[MOBILE ACCESS]
1. Make sure ComfyUI runs with --listen 0.0.0.0
   (Comfy Desktop: already enabled)
2. On your phone browser, go to:
   http://[YOUR_COMPUTER_IP]:8501
3. The IP address is shown when you start the server

[TROUBLESHOOTING]
- "API Offline" : Check Settings > ComfyUI Connection
  - Comfy Desktop users: use port 8189
  - Manual install: use port 8188
  - Make sure ComfyUI is actually running
- "No models found" : 
  - Use Settings > Manual Model Management to add by name
  - Or check if model directory is detected correctly
- Window flashes and closes : Run start.bat from command line
  to see error messages
- Password forgotten : Clear browser localStorage for this site

============================================================
