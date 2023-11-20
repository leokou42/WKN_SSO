from src.train import Train_pipeline
from src.test import Test_pipeline

hyper_parameter = [32, 50, 0.001]   # [batch_size, num_epochs, learning_rate]
Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
work_condition = 1
exp_name = 'noSSO_wc2_1st'

train_result = Train_pipeline(Learning_set, hyper_parameter, work_condition, exp_name)
print("PTH saved done!")

if work_condition == 1:
    Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_3',
                'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_4',
                'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_7']
elif work_condition == 2:
    Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_1',
                 'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_2']
elif work_condition == 3:
    Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing3_1',
                'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing3_2']

test_result = []

for test_set in Test_set:
    tmp_result = Test_pipeline(test_set, hyper_parameter[0], work_condition)
    test_result.append(tmp_result)

print("Finish !")