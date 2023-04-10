# Hand_Gesture_Controller

## introduction

以標準神經網路(NN)模型分別訓練及分析手勢,透過手指位置控制鼠標,並輔以各種手勢操控電腦之各項功能,例如按下左鍵、右鍵、雙擊等功能,進而達到無須滑鼠鍵盤也能操作電腦之目的。

## 操作說明
### Mouse mode
![image](https://user-images.githubusercontent.com/81171903/230874769-303025ba-d55b-4b82-8533-b69ac4e541ca.png)
![image](https://user-images.githubusercontent.com/81171903/230874913-9d8e9390-d9d0-4f26-9419-36cb709e1ed8.png)
## Keyboard mode
### 靜態操作
![image](https://user-images.githubusercontent.com/81171903/230874992-5068fa29-9458-428d-b943-5bc9ad3c478e.png)
![image](https://user-images.githubusercontent.com/81171903/230875053-c8fa0fa0-f355-42a2-b199-4a0d699fe87e.png)
![image](https://user-images.githubusercontent.com/81171903/230875095-9d2beaa8-f85e-4eb6-8823-287f773255c7.png)
### 動態操作 (利用LSTM實作)
![image](https://user-images.githubusercontent.com/81171903/230875382-422e390a-2378-474b-9090-09d50cedfd47.png)
# Rest mode
![image](https://user-images.githubusercontent.com/81171903/230875458-cc34a31e-c838-4e04-8c4d-b3e4fdfeb058.png)

## 更新log
1.	開啟程式會產生一個圓形泡泡，點擊run會出現opencv視窗
2.	圓形泡泡永遠置頂，可以隨意拖動，並以顏色區分mode
3.	Mouse mode => Blue, Rest mode => Red, Keyboard mode => Green
2.	10秒未使用進入rest mode (停止大部分偵測功能，僅使用6手勢切換mode)
3.	快轉五秒倒退五秒，持續姿勢超過三秒會加速成每0.25秒跳一次
4.	新增google小姐語音提醒
