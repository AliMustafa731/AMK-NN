
// enable windows visual theme style
#pragma comment(linker,"\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

#include <Windows.h>
#include <process.h>

#include <utils/random.h>
#include <neural_network.h>
#include <gui/program.h>
#include "loaders.h"

//-----------------------------------------------------
//    Declarations
//-----------------------------------------------------

const float PI = 3.1415926535897;
const float PI_2 = 6.28318530717;

// the nueral network structure
NeuralNetwork network;
MSELoss loss_function;
Adam optimizer;

// data
Tensor<float> netowrk_input, network_label;
Tensor<float> dataset, labels;

Image img;
Tensor<float> f_img;

// status
bool training = false;
const char* header_txt = "\nBy : Ali Mustafa Kamel\n2022-2023";

// windows specific variables
OPENFILENAME of_load_amknn = { 0 };
OPENFILENAME of_save_amknn = { 0 };
char file_name[512];

TrackBar TB_learn_rate;
TrackBar TB_momentum;
TrackBar TB_squared_grad;

HWND txt[32];
HDC win_hdc;

enum Control_ID
{
    TRAIN_BUTTON,
    LOAD_MODEL_BUTTON,
    SAVE_MODEL_BUTTON,
    RANDOM_SELECT_BUTTIN,
    NEXT_BUTTON,
    PREV_BUTTON
};

void train_network_thread(void *args);

struct MyApp : Program
{
    MyApp(){}

    void onCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
    {
        // initilize the neural network
        network.init(Shape(2, 2));
        network.add(new FullLayer(10));
        network.add(new SineLayer());

        optimizer = Adam(0.008f, 0.75f, 0.95f);

        netowrk_input.init(network.in_shape.size());
        network_label.init(network.output_layer()->out_size);
        loss_function.init(network.output_layer()->out_size);

        win_hdc = GetDC(hwnd);

        // load data

        if (load_mnist_images("mnist_digits_images_train.bin", dataset, 0) == false)
        {
            MessageBox(NULL, "Error : can;t load \"mnist_digits_images.bin\", make sure the executable is in it's main directory", "Opss!", MB_OK);
            PostQuitMessage(0);
        }

        f_img = dataset.slice({ dataset.shape[0], dataset.shape[1] }, { 0, 0, 0, rand32() % dataset.shape[3] });

        img = Image(dataset.shape[0], dataset.shape[1]);
        img.inti_from_float(f_img);
        img.draw(win_hdc, 32, 32, 256, 256);

        // create buttons & controls
        CreateWindow
        (
            WC_BUTTON, "Start training", WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
            625, 460, 120, 40, hwnd, (HMENU)TRAIN_BUTTON, GetModuleHandle(NULL), NULL
        );
        CreateWindow
        (
            WC_BUTTON, "Save model", WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
            800, 460, 120, 40, hwnd, (HMENU)SAVE_MODEL_BUTTON, GetModuleHandle(NULL), NULL
        );
        CreateWindow
        (
            WC_BUTTON, "Load model", WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
            800, 530, 120, 40, hwnd, (HMENU)LOAD_MODEL_BUTTON, GetModuleHandle(NULL), NULL
        );
        CreateWindow
        (
            WC_BUTTON, "Random Select", WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
            100, 350, 120, 40, hwnd, (HMENU)RANDOM_SELECT_BUTTIN, GetModuleHandle(NULL), NULL
        );

        txt[0] = CreateWindow
        (
            "STATIC", "0", WS_VISIBLE | WS_CHILD | SS_LEFT,
            780, 420, 150, 30, hwnd, NULL, GetModuleHandle(NULL), NULL
        );
        txt[1] = CreateWindow
        (
            "STATIC", "Reconstruction Loss : ", WS_VISIBLE | WS_CHILD | SS_LEFT,
            625, 420, 150, 30, hwnd, NULL, GetModuleHandle(NULL), NULL
        );
        txt[2] = CreateWindow
        (
            "STATIC", header_txt, WS_VISIBLE | WS_CHILD | SS_CENTER,
            615, 525, 150, 100, hwnd, NULL, GetModuleHandle(NULL), NULL
        );

        TB_learn_rate = TrackBar(hwnd, 75, 450, 300, 30, "0.0", "0.1  Learn Rate", 0.0f, 0.1f);
        TB_momentum = TrackBar(hwnd, 75, 500, 300, 30, "0.0", "1.0  Momentum");
        TB_squared_grad = TrackBar(hwnd, 75, 550, 300, 30, "0.0", "1.0  Squared Grad");

        TB_learn_rate.SetPos(optimizer.learning_rate);
        TB_momentum.SetPos(optimizer.beta1);
        TB_squared_grad.SetPos(optimizer.beta2);

        // Initialize OPENFILENAME for loading and saving

        CreateOPENFILENAME
        (
            &of_load_amknn, hwnd, file_name, sizeof(file_name), ".AMKnn Neural Network format\0*.AMKnn\0\0", "AMKnn",
            OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY
        );

        CreateOPENFILENAME
        (
            &of_save_amknn, hwnd, file_name, sizeof(file_name), ".AMKnn Neural Network format\0*.AMKnn\0\0", "AMKnn",
            OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT
        );
    }

    void onCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
    {
        int win_id = LOWORD(wParam);

        if (win_id == TRAIN_BUTTON)
        {
            training = !(training);

            if (training)
            {
                SetWindowText(GetDlgItem(hwnd, TRAIN_BUTTON), "Stop");

                EnableWindow(GetDlgItem(hwnd, SAVE_MODEL_BUTTON), FALSE);
                EnableWindow(GetDlgItem(hwnd, LOAD_MODEL_BUTTON), FALSE);

                _beginthread(train_network_thread, 0, NULL);
            }
            else
            {
                SetWindowText(GetDlgItem(hwnd, TRAIN_BUTTON), "Start training");

                EnableWindow(GetDlgItem(hwnd, SAVE_MODEL_BUTTON), TRUE);
                EnableWindow(GetDlgItem(hwnd, LOAD_MODEL_BUTTON), TRUE);
            }
        }

        if (win_id == SAVE_MODEL_BUTTON)
        {
            if (GetSaveFileName(&of_save_amknn))
            {
                std::ofstream file;
                file.open(file_name, std::ios::out | std::ios::binary);

                network.save(file);
                optimizer.save(file);

                file.close();
            }
        }

        if (win_id == LOAD_MODEL_BUTTON)
        {
            if (GetOpenFileName(&of_load_amknn))
            {
                std::ifstream file;
                file.open(file_name, std::ios::in | std::ios::binary);

                network.load(file);
                optimizer.load(file);

                file.close();

                TB_learn_rate.SetPos(optimizer.learning_rate);
                TB_momentum.SetPos(optimizer.beta1);
                TB_squared_grad.SetPos(optimizer.beta2);
            }
        }

        if (win_id == RANDOM_SELECT_BUTTIN)
        {
            // select a random sample image from the loaded dataset
            f_img = dataset.slice({ dataset.shape[0], dataset.shape[1] }, { 0, 0, 0, rand32() % dataset.shape[3] });

            img.inti_from_float(f_img);
            img.draw(win_hdc, 32, 32, 256, 256);
        }
    }

    void onDraw(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);

        FillRect(hdc, &ps.rcPaint, (HBRUSH)COLOR_WINDOW);

        img.draw(hdc, 32, 32, 256, 256);

        EndPaint(hwnd, &ps);
        ReleaseDC(hwnd, hdc);
    }

    void onHScroll(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
    {
        // update Trackbars
        TB_squared_grad.update();
        TB_learn_rate.update();
        TB_momentum.update();

        optimizer.learning_rate = TB_learn_rate.GetPos();
        optimizer.beta1 = TB_momentum.GetPos();
        optimizer.beta2 = TB_squared_grad.GetPos();
    }
};

//-----------------------------------------------------
//    Training & Logic of the program
//-----------------------------------------------------

// thread used to train the neural netwrok
void train_network_thread(void *args)
{
    int w = dataset.shape[0];
    int h = dataset.shape[1];
    float w_f = (float)w;
    float h_f = (float)h;
    int size = w * h;

    while (training)
    {
        float loss = 0;

        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h; y++)
            {
                network_label[0] = f_img[x + y * w] / 255.0f;
                // (x, y) are normalized in range (-PI, PI)
                netowrk_input[0] = ((float)x / w_f) * PI_2 - PI;
                netowrk_input[1] = ((float)y / h_f) * PI_2 - PI;

                Tensor<float>& network_out = network.forward(netowrk_input);
                network.backward(loss_function.gradient(network, network_label, size));

                float _loss = network_label[0] - network_out[0];
                loss += _loss * _loss * 0.5f;
            }
        }

        optimizer.update(network.parameters);
        SetWindowText(txt[0], std::to_string(loss).c_str());
    }
}

//-----------------------------------------------------
//    Main Entry
//-----------------------------------------------------

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, int nCmdShow)
{
    MyApp program;

    program.start("AMK Neural Network", 960, 640);

    return 0;
}
