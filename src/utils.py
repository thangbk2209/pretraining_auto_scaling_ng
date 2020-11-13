import matplotlib.pyplot as plt
from config import *


def plot_result(test_loss_list, y_true, y_pred, rmse_loss):
    save_dir = os.path.join(PLOT_RESULT_DIR, RUN_ID)
    try:
        os.mkdir(save_dir)
    except Exception as e:
        print(e)
        return None

    # plot_hist
    # plt.plot(train_loss_list)
    plt.plot(test_loss_list)
    plt.title('testing loss - Epoch')
    plt.legend(['testing_loss'])
    plt.savefig(os.path.join(save_dir, 'history'))

    # all
    plt.figure(figsize=(25, 10), linewidth=0.2)
    plt.plot(y_true.reshape(-1))
    plt.plot(y_pred.reshape(-1))
    plt.title('Test rmse={:.4f}'.format(rmse_loss))
    plt.legend(['True', 'Prediction'])
    plt.savefig(os.path.join(save_dir, f'true_pred_all.png'))

    # [:100]
    plt.figure(figsize=(25, 10), linewidth=0.2)
    plt.plot(y_true.reshape(-1)[:100])
    plt.plot(y_pred.reshape(-1)[:100])
    plt.title('Test rmse={:.4f}, 100 first'.format(rmse_loss))
    plt.legend(['True', 'Prediction'])
    plt.savefig(os.path.join(save_dir, f'true_pred_100first.png'))

    # [-100:]
    plt.figure(figsize=(25, 10), linewidth=0.2)
    plt.plot(y_true.reshape(-1)[-100:])
    plt.plot(y_pred.reshape(-1)[-100:])
    plt.title('Test rmse={:.4f}, 100 last'.format(rmse_loss))
    plt.legend(['True', 'Prediction'])
    plt.savefig(os.path.join(save_dir, f'true_pred_100last.png'))