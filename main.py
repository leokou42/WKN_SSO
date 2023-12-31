
import sys
sys.path.append("src")
import time 

from src.train import Train_pipeline
from src.test import Test_pipeline
from src.utils import output_2_csv, output_2_plot

# setup
hyper_parameter = [32, 50, 0.001]   # [batch_size, num_epochs, learning_rate]
batch_size = hyper_parameter[0]
Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
work_condition = [1,2,3]
exp_topic = 'noSSO'
exp_num = 1

# train
start_time1 = time.time()
for wc in work_condition:
    exp_name = exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st'
    train_result = Train_pipeline(Learning_set, hyper_parameter, wc, exp_name)
    print("{}, PTH saved done!".format(train_result))
end_time1 = time.time()
train_time = end_time1-start_time1

# test
start_time2 = time.time()
for wc in work_condition:
    trained_pth = 'F:/git_repo/WKN_SSO/result/pth/'+exp_topic+'_wc'+str(wc)+'_'+str(exp_num)+'st.pth'
    if wc == 1:
        Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_3',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_4',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_5',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_6',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing1_7']
    elif wc == 2:
        Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_3',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_4',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_5',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_6',
                    'F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing2_7']
    elif wc == 3:
        Test_set = ['F:/git_repo/WKN_SSO/viberation_dataset/Test_set/Bearing3_3']
        
    for test_set in Test_set:
        tmp_result = []
        bearing_name, tmp_result = Test_pipeline(trained_pth, test_set, batch_size, wc)
        output_2_csv(bearing_name, tmp_result)
        output_2_plot(bearing_name, tmp_result)
end_time2 = time.time()
test_time = end_time2-start_time2

print("Finish !")
print("Train Time = {}, Test Time = {}".format(train_time, test_time))