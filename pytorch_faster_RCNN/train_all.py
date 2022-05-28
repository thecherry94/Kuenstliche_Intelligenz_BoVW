import os
from datetime import datetime, date

def main():
    categories = next(os.walk('../mvtec_anomaly_detection_data/'))[1]
    
    num_epochs = 5
    learning_rate = 0.002
    
    t1 = datetime.now()
    
    log = "Training on " + str(categories) + " categories with " + str(num_epochs) + " epochs and " + str(learning_rate) + " learning rate"
    
    for cat in categories:
        t1_sub = datetime.now()
        os.system(f'python train.py -category {cat} --epochs {num_epochs} --lr {learning_rate}')
        t2_sub = datetime.now()
        
        dt_sub = t2_sub - t1_sub
        
        hours = int(dt_sub.total_seconds() / 3600)
        minutes = int((dt_sub.total_seconds() % 3600) / 60)
        seconds = int(dt_sub.total_seconds() % 60)
        
        msg = f'Training of {cat} took {hours} hours, {minutes} minutes, {seconds} seconds'
        log += msg + '\n'
        print(msg)
        
    
    t2 = datetime.now()
    
    dt = t2 - t1
    
    hours = int(dt.total_seconds() / 3600)
    minutes = int((dt.total_seconds() % 3600) / 60)
    seconds = int(dt.total_seconds() % 60)
    
    msg = '---------------------------------------------------------------\n'
    msg += f'Training of all categories took {hours} hours, {minutes} minutes, {seconds} seconds\n'
    msg += '---------------------------------------------------------------'
    
    log += msg
    print(msg)
    
    print('Writing log file')
    with open('log.txt', 'w') as f:
        f.write(log)
    
if __name__ == '__main__':
    main()