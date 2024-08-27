## Commands to run on production

RAG_for_app is being used by api.py on a server to run continuously as an api endpoint.

The server is on 152.71.142.174
The process is running on port 8001, can be accessed via an ngrok tunnel:
> virtually-talented-monster.ngrok-free.app

#### Command to run api:
After activating the venv on the server (that uses packages in requirements-app.txt)
uvicorn api:app --host 0.0.0.0 --port 8001
(on path: /home/akash/HSv2 )

#### Command on server to run ngrok:
ngrok http --domain=virtually-talented-monster.ngrok-free.app 8001

(in case of aut issues:
ngrok config add-authtoken 2lFEhI6RiixGMYVYCnDACOEDGMS_9Zxad9ssiC73tKyaVygH)


## Deployment info
### Screen info
Fastapi and ngrok are running in a screen process (two windows)
- Attach to the screen:
`screen -r hsv2`
- Switch windows:
`Ctrl + A`, then `N`
- Detach from screen
`Ctrl + A`, then `D`

### Deployed API
Access api here:
https://virtually-talented-monster.ngrok-free.app/

To talk to the llm:
https://virtually-talented-monster.ngrok-free.app/llm?input=Hello how are you
(Use the query parameter input to provide user input)