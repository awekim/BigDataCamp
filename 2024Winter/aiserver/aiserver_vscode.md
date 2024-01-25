0. Download key file from server & run following code
>>> chmod 600 ai-server_key.pm

1. Add SSH config
Host Azure-server2
        HostName 20.200.221.5
        User azureuser
        IdentityFile "/Users/keungouikim/Desktop/UrgentTask/ai_server/ai-server_key.pem"

2. Open folder (SSH)

3. Add files

4. Open VSCode terminal and run following codes
>>> sudo apt-get update
>>> sudo apt-get install python3-pip
>>> sudo apt install python3.8-venv
>>> cd workspace
>>> python3 -m venv myenv
>>> source ./myenv/bin/activate
>>> pip install pandas
>>> pip install scikit-learn
>>> pip install numpy
>>> pip install keras
>>> pip install tensorflow
>>> pip install fastapi
>>> pip install uvicorn

5. Run following code
>>> python bitcoin.py

6. Run following code (run ai server)
uvicorn --host 0.0.0.0 --port 8080 api:app --reload
(--host 0.0.0.0: --host 옵션은 서버가 들어오는 요청을 수신할 주소를 지정합니다. '0.0.0.0'을 사용하면 서버가 컴퓨터의 사용 가능한 모든 IP 주소를 수신하여 네트워크의 다른 컴퓨터에서 또는 적절하게 구성된 경우 인터넷에서 액세스할 수 있게 됩니다.
--port 8080: 서버가 수신할 포트를 설정합니다. 이 경우 8080으로 설정되어 있으므로 http://localhost:8080 또는 http://[컴퓨터의 IP 주소]:8080에서 웹 애플리케이션에 액세스하게 됩니다.

api:app: 이 부분은 Uvicorn이 실행해야 하는 Python 모듈과 ASGI 애플리케이션 인스턴스를 지정합니다. 여기서 api는 Python 파일의 이름(.py 확장자 없음)이고 app은 해당 파일 내의 ASGI 애플리케이션 인스턴스의 이름입니다.

--reload: 선택적 인수입니다. 포함된 경우 소스 코드의 변경 사항을 감지하면 Uvicorn에 서버를 자동으로 다시 시작하도록 지시합니다. 이는 코드를 변경할 때마다 서버를 수동으로 다시 시작하지 않아도 되므로 개발 중에 매우 유용합니다.)

7. Open another terminal and run following code to check the performance
curl -X POST -H "Content-Type: application/json" -d "{\"closing_price\": \"500\", \"opening_price\": \"400\", \"high_price\": \"400\", \"low_price\": \"400\", \"trading_volume\": \"4000000\"}" http://20.200.221.5:8080
