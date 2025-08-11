# How to configure the launch.json in multi-gpu training launch.py
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Python Debugger: Current File with Arguments",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/training/launch.py",
            "cwd": "${workspaceFolder}/training", 
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",   // 只用 0 号卡
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500"
            },
            
        }
    ]
}

```
