# drone

[22.08.11]

라우터에 연결된 기기는 외부 ip가 고정되어 일정한 ip를 가지므로 LCD를 통해 접속 주소를 나타내는 것은 불필요할 것으로 판단하여 제거
라즈베리파이 내부 ip는 정적 ip로 설정하는 것이 요구됨

네트워크 접속 속도 개선을 위해 라우터-라즈베리파이의 연결이 와이파이에서 랜선으로 바꾸는 것을 생각해 봐야함
실험 하고 유의미한 속도차이가 있다면 랜선으로 변경함

라즈베리파이의 연산속도를 time.sleep으로 조정하여 라즈베리파이의 cpu 부하를 줄여보고 조종에 문제가 될만큼 코드에 delay를 줘야 한다면
라즈베리파이의 기능을 분리한다.
1. 라즈베리파이는 카메라 영상만을 송출한다. [drone_camera_only.py 를 실행]
 --> 라우터 고정 ip 주소:8080
2. 숫자 인식 코드를 실행할 노트북을 준비한다.
 --> cap.VideoCapture('라우터 고정 ip 주소:8080')
 위 코드를 제외한 나머지 전체 코드 (drone_main.py) 과 같게 한다.
위 방법이 가장 확실하고 좋은 품질의 영상을 얻을 수 있는 방법이 될 것으로 예상된다.
