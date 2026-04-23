from fastapi import FastAPI
import uvicorn
import sys

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "alive"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    # Get port from command line or default to 8765
    port = 8765
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    
    uvicorn.run(app, host="127.0.0.1", port=port)
