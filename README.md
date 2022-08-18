# 경북대학교 기계공학부 천상연 2022년 회전익(드론) 팀
### [2022년] 14회 부산대학교 총장배 창의비행체 경진대회 경북대 포트리스팀 임무장비 코드
▶임무 내용: 타종, 숫자 인식하여 해당 숫자의 착륙지에 착륙

▶제작 영상: https://youtu.be/fozkv8buq-I

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
 
 위 코드를 제외한 나머지 전체 코드 (drone_main.py) 과 같게 한다.  해당 방법이 가장 확실하고 좋은 품질의 영상을 얻을 수 있는 방법이 될 것으로 예상된다.

[22.08.19]

라즈베리파이의 연산속도보다 네트워크 상태가 영상의 끊김에 더 큰 영향을 미침
time.sleep으로 조정 시 네트워크에 의한 끊김, cpu 스로틀링에 의한 끊김에 더해 명령어로 인한 딜레이가 겹처 더 끊기는 것 같음

숫자에 틈이 있을 경우 보정을 하도록 해뒀었는데 해당 코드는 옵션 선택을 해야만 작동하도록 변경하여 평상시 연산량을 
