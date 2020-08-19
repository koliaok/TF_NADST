"""
No GATE = 0 
Average latency: 0.026443275061454885
Joint Gate Acc 0.0
Joint Lenval Acc 0.6083061889250815
Joint Acc 0.4824918566775244 Slot Acc 0.9724329145339253 F1 0.8858202074051936
Oracle Joint Acc 0.7064332247557004 Slot Acc 0.9458732663959895 F1 0.9886846595315982
"""

"""
1차 모델 실험(Decoder의 같은 Attention)
GATE = 0
Epoch 8 gate loss -0.0000 lenval loss 0.0002 state loss 1.4254
 joint_gate acc 0.0000 joint_lenval acc 0.6188 joint acc 0.6258 f1 0.9161 turn acc 0.9828

2차 모델 실험(Attention 따로)
Epoch 33 gate loss -0.0000 lenval loss 0.0002 state loss 1.4041
 joint_gate acc 0.0000 joint_lenval acc 0.6156 joint acc 0.6516 f1 0.9270 turn acc 0.9848

"""