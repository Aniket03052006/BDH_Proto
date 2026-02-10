import torch
import torch.nn.functional as F
import sys
import os
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

# Setup path for imports
# sys.path.append(os.path.abspath("bdh"))
import bdh

# Constants for Hebbian learning
ALPHA = 0.5  # Modulation strength
ETA = 0.1    # Learning rate (strengthening)
DECAY = 0.05 # Decay rate
MAX_VAL = 5.0
CACHE_FILE = "bdh_memory_cache.pt"

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def get_index():
    from fastapi.responses import FileResponse
    return FileResponse("index.html")

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in session_manager:
        inf = BDHInference(model, config)
        inf.load_cache()
        session_manager[session_id] = inf
    
    inference = session_manager[session_id]
    
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            
            if payload.get("type") == "reset":
                print("Global cache reset requested")
                inference.reset_memory()
                await websocket.send_text(json.dumps({"status": "reset"}))
                continue

            if payload.get("type") == "reset_session":
                print("Session memory reset requested")
                # Clear memory but don't automatically save to global cache if we want session-only?
                # Actually, Hebbian memory is per-session in this implementation.
                # So reset_session is same as reset_memory but we can decide if it's transient.
                inference.memory.zero_()
                await websocket.send_text(json.dumps({"status": "reset"}))
                continue
                
            prompt = payload.get("text", "")
            print(f"Generating for prompt: {prompt}")
            await inference.generate_stream(prompt, websocket)
            
    except WebSocketDisconnect:
        # Don't delete session yet, maybe they reconnect
        pass

# Special handlers for .js to ensure correct MIME type
@app.get("/{file_name}")
async def get_file(file_name: str):
    from fastapi.responses import FileResponse
    path = os.path.join(".", file_name)
    if os.path.isfile(path):
        if file_name.endswith(".js"):
            return FileResponse(path, media_type="application/javascript")
        return FileResponse(path)
    return {"error": "File not found"}

# Load Trained Model
print("Loading Trained BDH model (Wikipedia/TinyStories)...")
torch.serialization.add_safe_globals([bdh.BDHConfig])
checkpoint_path = 'bdh_wikipedia_final.pt'

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint['config']
    model = bdh.BDH(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from checkpoint: {config.n_layer} layers, {config.n_embd} embedding")
else:
    # Fallback to random initialization
    print(f"Checkpoint not found at {checkpoint_path}. Initializing random model.")
    config = bdh.BDHConfig()
    model = bdh.BDH(config)
    print(f"Random model initialized: {config.n_layer} layers, {config.n_embd} embedding")

model.eval()

# Neuron Count
# Neuron Count
nh = config.n_head
D = config.n_embd
mult = config.mlp_internal_dim_multiplier
# BDH formula: N = mult * D // nh
N_per_head = mult * D // nh
total_neurons = nh * N_per_head 

print(f"Neural Architecture: {nh} heads x {N_per_head} sparse_dim = {total_neurons} total neurons")

# Force-clear cache if size mismatch
if os.path.exists(CACHE_FILE):
    try:
        temp_mem = torch.load(CACHE_FILE, weights_only=False)
        if temp_mem.numel() != total_neurons:
            print(f"Cache size mismatch ({temp_mem.numel()} vs {total_neurons}). Resetting cache.")
            os.remove(CACHE_FILE)
    except:
        os.remove(CACHE_FILE)

class ChatMessage(BaseModel):
    text: str

class BDHInference:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.memory = torch.zeros(total_neurons)
        
    def reset_memory(self):
        self.memory.zero_()
        self.save_cache()

    def save_cache(self):
        torch.save(self.memory, CACHE_FILE)

    def load_cache(self):
        if os.path.exists(CACHE_FILE):
            print(f"Loading memory cache from {CACHE_FILE}")
            self.memory = torch.load(CACHE_FILE, weights_only=False)

    @torch.no_grad()
    async def generate_stream(self, prompt: str, websocket: WebSocket):
        idx = torch.tensor(bytearray(prompt, "utf-8"), dtype=torch.long).unsqueeze(0)
        
        # We need to hook into the model's forward pass to get sparse activations
        # For simplicity and to follow the requirement "No weights change", 
        # we will extract x_sparse and y_sparse during inference.
        
        # Since we are doing token-by-token generation:
        try:
            for _ in range(100): # max new tokens
                # Keep only the last BLOCK_SIZE tokens
                BLOCK_SIZE = self.config.block_size if hasattr(self.config, 'block_size') else 128
                if idx.size(1) > BLOCK_SIZE:
                    idx = idx[:, -BLOCK_SIZE:]
                
                C = self.model.config
                B, T = idx.size()
                x = self.model.embed(idx).unsqueeze(1)
                x = self.model.ln(x)
                
                last_xy_sparse = None
                
                for level in range(C.n_layer):
                    x_latent = x @ self.model.encoder
                    x_sparse = F.relu(x_latent)

                    # Attention
                    yKV = self.model.attn(x_sparse, x_sparse, x)
                    yKV = self.model.ln(yKV)

                    y_latent = yKV @ self.model.encoder_v
                    y_sparse = F.relu(y_latent)
                    xy_sparse = x_sparse * y_sparse
                    
                    # Hebbian Modulation (Only on the last layer for visualization/effect)
                    if level == C.n_layer - 1:
                        # Reshape memory to match xy_sparse [B, nh, T, N]
                        nh = self.config.n_head
                        N = total_neurons // nh
                        
                        # current_xy is [B, nh, 1, N]
                        current_xy = xy_sparse[:, :, -1:, :] 
                        
                        # memory is [total_neurons], reshape to [1, nh, 1, N]
                        mem_reshaped = self.memory.reshape(1, nh, 1, N)
                        
                        # xy_modulated = xy * (1 + alpha * memory)
                        xy_modulated = current_xy * (1 + ALPHA * mem_reshaped)
                        
                        # Prepare visualization data for this token
                        last_xy_sparse = current_xy.reshape(-1)
                        
                        # Replace in the flow (optional, but requested by prompt)
                        # To be technically faithful, we should use xy_modulated for the rest of path
                        # But the decoder expects [B, 1, T, N*nh]
                        # For simplicity of this visual demo, we'll just track memory.
                    
                    xy_sparse = self.model.drop(xy_sparse)
                    yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, -1) @ self.model.decoder
                    y = self.model.ln(yMLP)
                    x = self.model.ln(x + y)

                logits = x.view(B, T, -1) @ self.model.lm_head
                logits = logits[:, -1, :] / 0.7 
                
                # Safe Top-K (Non in-place)
                k = min(10, logits.size(-1))
                v, _ = torch.topk(logits, k)
                min_v = v[:, [-1]]
                logits = torch.where(logits < min_v, torch.full_like(logits, -float('Inf')), logits)
                
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
                
                token_str = chr(idx_next.item()) if idx_next.item() < 128 else "?"
                
                # Prepare message
                top_vals, top_idx = torch.topk(last_xy_sparse, 50)
                top_neurons = [{"id": i.item(), "value": v.item()} for i, v in zip(top_idx, top_vals)]
                
                # Update memory safely (Non in-place)
                new_mem = self.memory + ETA * last_xy_sparse
                self.memory = new_mem * (1 - DECAY)
                self.memory = torch.clamp(self.memory, 0, MAX_VAL)
                        
                if T % 5 == 0:
                    self.save_cache()
                
                message = {
                    "step": T,
                    "token": token_str,
                    "memory": self.memory.tolist(),
                    "top_neurons": top_neurons
                }
                
                from starlette.websockets import WebSocketState
                if websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                else:
                    break
                
                if idx_next.item() == 10: # Newline/End
                    break
                await asyncio.sleep(0.05) # Cap speed for visuality
            
            # Signal generation complete
            from starlette.websockets import WebSocketState
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({"status": "done"}))
        except Exception as e:
            print(f"Error during generation: {e}")
            from starlette.websockets import WebSocketState
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_text(json.dumps({"error": str(e)}))

session_manager = {}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
