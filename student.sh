python main_student.py \
  --dataset ogbg-molhiv \
  --model_name=Graph_Student_MSE \
  --by_default \
  --gnn=gcn-virtual \
  --device=3 \
  --trails=5 \
  --beta_infonce=0.01 \
  --beta_club=0.01 \
  --train_type=student \
  --teacher_model=./ogbg-molbace_gcn_teacher_3 \



  # molsider
  # molbbbp
  # molbace
  # molhiv
  # moltoxcast
