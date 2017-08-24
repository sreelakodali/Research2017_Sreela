%sub1(sub1==16)=10
%sub2(sub2==16)=10
%sub3(sub3==16)=10
%Sub4(Sub4==16)=10
%sub5(sub5==16)=10
%sub6(sub6==16)=10
%sub7(sub7==16)=10
%sub8(sub8==16)=10
%sub9(sub9==16)=10

%sub1(sub1==17)=11
%sub2(sub2==17)=11
%sub3(sub3==17)=11
%Sub4(Sub4==17)=11
%sub5(sub5==17)=11
%sub6(sub6==17)=11
%sub7(sub7==17)=11
%sub8(sub8==17)=11
%sub9(sub9==17)=11

%sub1(sub1==24)=12
%sub2(sub2==24)=12
%sub3(sub3==24)=12
%Sub4(Sub4==24)=12
%sub5(sub5==24)=12
%sub6(sub6==24)=12
%sub7(sub7==24)=12
%sub8(sub8==24)=12
%sub9(sub9==24)=12

%sub2 = sub2 + 1
%sub3 = sub3 + 1
%Sub4 = Sub4 + 1
%sub5 = sub5 + 1
%sub6 = sub6 + 1
%sub7 = sub7 + 1
%sub8 = sub8 + 1
%sub9 = sub9 + 1

dsinputs_testPAMAP = downsample(inputs_testPAMAP, 3)
dsinputs_trainingPAMAP = downsample(inputs_trainingPAMAP, 3)
dsinputs_valPAMAP = downsample(inputs_valPAMAP, 3)

dstargets_testPAMAP = downsample(targets_testPAMAP, 3)
dstargets_trainingPAMAP = downsample(targets_trainingPAMAP, 3)
dstargets_valPAMAP = downsample(targets_valPAMAP, 3)