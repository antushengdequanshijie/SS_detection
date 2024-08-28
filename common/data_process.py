import pandas as pd
from datetime import datetime

def get_label_info(excel_path):
    # excel_path = "../2.xlsx"
    pd.set_option('display.max_rows', None)
    df = pd.read_excel(excel_path, header=2)
    # 先创建一个完整的 DataFrame 副本
    data_time_df = df[['数据', '记录时间']].copy()
    # 去除可能的空格
    data_time_df['记录时间'] = data_time_df['记录时间'].str.strip()  # 去除空格很重要
    # 明确指定日期时间格式
    data_time_df['记录时间'] = pd.to_datetime(data_time_df['记录时间'], errors='coerce')
    # 检查转换后的数据类型
    print("Data type after conversion with specific format:", data_time_df['记录时间'].dtype)
    print(data_time_df['记录时间'].head())
    return data_time_df
def extract_timestamp_from_filename(filename):
    # 提取文件名中的时间戳
    # timestamp_str = filename.split('_')[1].split('.')[0]
    timestamp_str = filename.split('_')[1][:12]
    # 将时间戳转换为datetime对象
    file_datetime = datetime.strptime(timestamp_str, '%Y%m%d%H%M')
    return file_datetime

def get_video_label(video_name,data_time_df):
    file_datetime = extract_timestamp_from_filename(video_name)
    matched_row = data_time_df[data_time_df['记录时间'] == file_datetime]
    # 检查匹配结果并提取数据
    if not matched_row.empty:
        # 假设“数据”列包含需要的信息
        data_value = matched_row['数据'].values[0]
        return data_value
        # print("Matching Data:", data_value)
    else:
        # print("No matching data found for the given timestamp.")
        return None


if __name__ == '__main__':
    video_name = "Video_20240703163349387.avi"
    data_time_df = get_label_info("../../2.xlsx")
    label = get_video_label(video_name, data_time_df)
