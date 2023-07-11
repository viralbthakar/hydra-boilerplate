def styled_print(text, header=False):
    """Custom Print Function"""
    class style:
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

    if header:
        print(f'{style.BOLD}› {style.UNDERLINE}{text}{style.END}')
    else:
        print(f'    {text}')

def split_data(x_train, y_train, training_observations_cnt):
    dat_offset = x_train.shape[0] - training_observations_cnt
    x_val = x_train[-dat_offset:]
    y_val = y_train[-dat_offset:]
    x_train = x_train[:-dat_offset]
    y_train = y_train[:-dat_offset]

    return (x_train, y_train), (x_val, y_val)