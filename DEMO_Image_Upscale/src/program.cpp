
#include <Windows.h>
#include <CommCtrl.h>
#include <fstream>
#include <string>
#include <ctime>

#include <process.h>
#include "program.h"
#include "loaders.h"

#include "data/dataset.h"
#include "utils/graphics.h"
#include "utils/utils.h"
#include "neural_network.h"

// enable windows visual theme style
#pragma comment(linker,"\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

//-----------------------------------------------------
//    All code that controls and manage the window
//-----------------------------------------------------

void Program::init(const char* name, int _w, int _h)
{
    srand(time(0)); // seed the random generator

    // create and register the window
    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = "AMKNN";
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    RegisterClass(&wc);

    win_handle = CreateWindowEx
    (
        0, "AMKNN", name,
        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
        CW_USEDEFAULT, CW_USEDEFAULT,
        _w, _h, NULL, NULL, GetModuleHandle(NULL), NULL
    );

    if (win_handle == NULL)
    {
        MessageBox(NULL, "Error : can't initialize the program : \"win_handle\"", "Opss!", MB_OK);
        exit(0);
    }

    ShowWindow(win_handle, SW_SHOW);
    UpdateWindow(win_handle);

    INITCOMMONCONTROLSEX icex;
    icex.dwSize = sizeof(INITCOMMONCONTROLSEX);
    icex.dwICC = ICC_LISTVIEW_CLASSES;
    InitCommonControlsEx(&icex);

    // run messages loop
    MSG msg = {};
    while (GetMessage(&msg, NULL, 0, 0))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

//-----------------------------------------------------
//    Main Window Callback
//-----------------------------------------------------

void onCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void onCommand(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void onDraw(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
void updateTrackbars(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_CREATE:
    {
        onCreate(hwnd, uMsg, wParam, lParam);
        updateTrackbars(hwnd, uMsg, wParam, lParam);

    } return 0;

    case WM_COMMAND:
    {
        onCommand(hwnd, uMsg, wParam, lParam);

    } return 0;

    case WM_HSCROLL:
    {
        updateTrackbars(hwnd, uMsg, wParam, lParam);

    } return 0;

    case WM_PAINT:
    {
        onDraw(hwnd, uMsg, wParam, lParam);

    } return 0;

    case WM_DESTROY:
    {
        PostQuitMessage(0);

    } return 0;

    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

//-----------------------------------------------------
//    Training & Logic of the program
//-----------------------------------------------------

DataSet data, labels;
Image img_1, img_2, img_3;
Buffer<float> f_img_1;

Array<float> netowrk_input, network_label;

NeuralNetwork drawer_network;
float learn_rate = 0.008f;
float momentum = 0.75f;
float squared_grad = 0.95f;
bool training = false;

const char* header_txt = "\nBy : Ali Mustafa Kamel\n2022-2023";

OPENFILENAME of_load_amknn = { 0 };
OPENFILENAME of_save_amknn = { 0 };
OPENFILENAME of_save_image = { 0 };
char file_name[512];

TrackBar treackbar_learn_rate;
TrackBar treackbar_momentum;
TrackBar treackbar_squared_grad;

HWND txt[32];
HDC win_hdc;

#define TRAIN_BUTTON 0
#define LOAD_MODEL_BUTTON 1
#define SAVE_MODEL_BUTTON 2
#define UP_SCALE_BUTTON 3
#define LOAD_BAR 4
#define RANDOM_SELECT_BUTTIN 5

void DrawerNetworkDraw(Buffer<Color> &dest, int w, int h)
{
    float w_f = (float)w;
    float h_f = (float)h;

    for (int x = 0; x < w; x++)
    {
        for (int y = 0; y < h; y++)
        {
            netowrk_input[0] = (float)x * 10.0f / w_f;
            netowrk_input[1] = (float)y * 10.0f / h_f;
            float *o = drawer_network.forward(netowrk_input.data);

            int idx = x + y * w;
            dest[idx].r = (unsigned char)(55.0f + o[0] * 200.0f);
            dest[idx].g = (unsigned char)(55.0f + o[0] * 200.0f);
            dest[idx].b = (unsigned char)0;
        }
    }
}

void up_scale_thread(void *args)
{
    DrawerNetworkDraw(img_3.img, 256, 256);
    img_3.draw(win_hdc, 632, 32, 256, 256);

    EnableWindow(GetDlgItem((HWND)args, UP_SCALE_BUTTON), TRUE);
    EnableWindow(GetDlgItem((HWND)args, SAVE_MODEL_BUTTON), TRUE);
    EnableWindow(GetDlgItem((HWND)args, LOAD_MODEL_BUTTON), TRUE);
    EnableWindow(GetDlgItem((HWND)args, TRAIN_BUTTON), TRUE);
}

void train_network_thread(void *args)
{
    int w = data.shape.w;
    int h = data.shape.h;
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
                network_label[0] = f_img_1[x + y * w] / 255.0f;
                netowrk_input[0] = (float)x * 10.0f / w_f;
                netowrk_input[1] = (float)y * 10.0f / h_f;

                float *o = drawer_network.forward(netowrk_input.data);
                MSELoss(&drawer_network, network_label.data, size);
                drawer_network.backward();

                float _loss = network_label[0] - o[0];
                loss += _loss * _loss * 0.5f;

                int idx = x + y * w;
                img_2.img[idx].r = (unsigned char)0;
                img_2.img[idx].g = (unsigned char)(55.0f + o[0] * 200.0f);
                img_2.img[idx].b = (unsigned char)(55.0f + o[0] * 200.0f);
            }
        }

        drawer_network.optimizer->update(drawer_network.parameters);
        img_2.draw(win_hdc, 332, 32, 256, 256);
        SetWindowText(txt[0], std::to_string(loss).c_str());
    }
}

void onCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    drawer_network.init
    (
        Shape(2, 1, 1),
        {
            new FullLayer(15, 0.0001f),
            new SineLayer(),
            new FullLayer(10, 0.0001f),
            new SineLayer(),
            new FullLayer(10, 0.0001f),
            new SineLayer(),
            new FullLayer(1),
            new SigmoidLayer()
        },
        new Adam(learn_rate, momentum, squared_grad)
    );

    netowrk_input.init(drawer_network.in_shape.size());
    network_label.init(drawer_network.output_layer()->out_size);

    win_hdc = GetDC(hwnd);

    // load data

    bool result_1 = load_mnist_images("mnist_digits_images.bin", data, 0);

    if (!result_1)
    {
        MessageBox(NULL, "Error : can;t load \"mnist_digits_images.bin\", make sure the executable is in it's main directory", "Opss!", MB_OK);
        PostQuitMessage(0);
    }

    img_1 = Image(data.shape.w, data.shape.h);
    img_2 = Image(data.shape.w, data.shape.h);
    img_3 = Image(256, 256);

    f_img_1.init(data.shape.w, data.shape.h, data[rand32() % data.samples_num]);
    embed_one_channel_to_color(img_1.img.data, f_img_1.data, data.sample_size);

    DrawerNetworkDraw(img_2.img, 28, 28);
    img_2.draw(win_hdc, 332, 32, 256, 256);

    DrawerNetworkDraw(img_3.img, 256, 256);
    img_3.draw(win_hdc, 332, 32, 256, 256);

    // create buttons & controls
    CreateWindow
    (
        WC_BUTTON, "Start training", WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        410, 350, 120, 40, hwnd, (HMENU)TRAIN_BUTTON, GetModuleHandle(NULL), NULL
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
        WC_BUTTON, "Up Scale", WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        720, 350, 120, 40, hwnd, (HMENU)UP_SCALE_BUTTON, GetModuleHandle(NULL), NULL
    );
    CreateWindow
    (
        WC_BUTTON, "Random Select", WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
        100, 350, 120, 40, hwnd, (HMENU)RANDOM_SELECT_BUTTIN, GetModuleHandle(NULL), NULL
    );

    txt[0] = CreateWindow
    (
        "STATIC", "0", WS_VISIBLE | WS_CHILD | SS_LEFT,
        532, 310, 150, 30, hwnd, NULL, GetModuleHandle(NULL), NULL
    );
    txt[1] = CreateWindow
    (
        "STATIC", "Reconstruction Loss : ", WS_VISIBLE | WS_CHILD | SS_LEFT,
        360, 310, 150, 30, hwnd, NULL, GetModuleHandle(NULL), NULL
    );
    txt[2] = CreateWindow
    (
        "STATIC", header_txt, WS_VISIBLE | WS_CHILD | SS_CENTER,
        615, 525, 150, 100, hwnd, NULL, GetModuleHandle(NULL), NULL
    );

    treackbar_learn_rate = TrackBar(hwnd, 75, 450, 300, 30, "0.0", "0.1  Learn Rate", 0.0f, 0.1f);
    treackbar_momentum = TrackBar(hwnd, 75, 500, 300, 30, "0.0", "1.0  Momentum");
    treackbar_squared_grad = TrackBar(hwnd, 75, 550, 300, 30, "0.0", "1.0  Squared Grad");

    treackbar_learn_rate.set_pos(learn_rate);
    treackbar_momentum.set_pos(momentum);
    treackbar_squared_grad.set_pos(squared_grad);

    // Initialize OPENFILENAME for loading and saving
    ZeroMemory(&of_load_amknn, sizeof(of_load_amknn));
    ZeroMemory(&of_save_amknn, sizeof(of_save_amknn));
    ZeroMemory(&of_save_image, sizeof(of_save_image));

    of_load_amknn.lStructSize = sizeof(of_load_amknn);
    of_load_amknn.hwndOwner = hwnd;
    of_load_amknn.lpstrFile = file_name;
    of_load_amknn.lpstrFile[0] = '\0';  // Set to '\0' so that GetOpenFileName does not use the contents of file_name to initialize itself.
    of_load_amknn.nMaxFile = sizeof(file_name);
    of_load_amknn.lpstrFilter = ".AMKnn Neural Network format\0*.AMKnn\0\0";
    of_load_amknn.nFilterIndex = 1;
    of_load_amknn.lpstrFileTitle = NULL;
    of_load_amknn.nMaxFileTitle = 0;
    of_load_amknn.lpstrInitialDir = NULL;
    of_load_amknn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;

    of_save_amknn.lStructSize = sizeof(of_save_image);
    of_save_amknn.hwndOwner = hwnd;
    of_save_amknn.lpstrFile = file_name;
    of_save_amknn.lpstrFile[0] = '\0';
    of_save_amknn.nMaxFile = sizeof(file_name);
    of_save_amknn.lpstrFilter = ".AMKnn Neural Network format\0*.AMKnn\0\0";
    of_save_amknn.nFilterIndex = 1;
    of_save_amknn.lpstrFileTitle = NULL;
    of_save_amknn.nMaxFileTitle = 0;
    of_save_amknn.lpstrInitialDir = NULL;
    of_save_amknn.lpstrDefExt = "AMKnn";
    of_save_amknn.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;

    of_save_image.lStructSize = sizeof(of_save_image);
    of_save_image.hwndOwner = hwnd;
    of_save_image.lpstrFile = file_name;
    of_save_image.lpstrFile[0] = '\0';
    of_save_image.nMaxFile = sizeof(file_name);
    of_save_image.lpstrFilter = ".png format\0*.png\0\0";
    of_save_image.nFilterIndex = 1;
    of_save_image.lpstrFileTitle = NULL;
    of_save_image.nMaxFileTitle = 0;
    of_save_image.lpstrInitialDir = NULL;
    of_save_image.lpstrDefExt = "png";
    of_save_image.Flags = OFN_PATHMUSTEXIST | OFN_OVERWRITEPROMPT;
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
            EnableWindow(GetDlgItem(hwnd, UP_SCALE_BUTTON), FALSE);

            _beginthread(train_network_thread, 0, NULL);
        }
        else
        {
            SetWindowText(GetDlgItem(hwnd, TRAIN_BUTTON), "Start training");

            EnableWindow(GetDlgItem(hwnd, SAVE_MODEL_BUTTON), TRUE);
            EnableWindow(GetDlgItem(hwnd, LOAD_MODEL_BUTTON), TRUE);
            EnableWindow(GetDlgItem(hwnd, UP_SCALE_BUTTON), TRUE);
        }
    }

    if (win_id == UP_SCALE_BUTTON)
    {
        EnableWindow(GetDlgItem(hwnd, UP_SCALE_BUTTON), FALSE);
        EnableWindow(GetDlgItem(hwnd, SAVE_MODEL_BUTTON), FALSE);
        EnableWindow(GetDlgItem(hwnd, LOAD_MODEL_BUTTON), FALSE);
        EnableWindow(GetDlgItem(hwnd, TRAIN_BUTTON), FALSE);

        _beginthread(up_scale_thread, 0, (void*)hwnd);
    }

    if (win_id == SAVE_MODEL_BUTTON)
    {
        if (GetSaveFileName(&of_save_amknn))
        {
            drawer_network.save(file_name);
        }
    }

    if (win_id == LOAD_MODEL_BUTTON)
    {
        if (GetOpenFileName(&of_load_amknn))
        {
            drawer_network.load(file_name);

            Adam* p = (Adam*)drawer_network.optimizer;
            learn_rate = p->learning_rate;
            momentum = p->beta1;
            squared_grad = p->beta2;

            treackbar_learn_rate.set_pos(learn_rate);
            treackbar_momentum.set_pos(momentum);
            treackbar_squared_grad.set_pos(squared_grad);

            DrawerNetworkDraw(img_2.img, 28, 28);
            img_2.draw(win_hdc, 332, 32, 256, 256);
        }
    }

    if (win_id == RANDOM_SELECT_BUTTIN)
    {
        int idx = rand32() % data.samples_num;
        f_img_1.data = data[idx];
        f_img_1.size = data.sample_size;
        embed_one_channel_to_color(img_1.img.data, f_img_1.data, data.sample_size);

        img_1.draw(win_hdc, 32, 32, 256, 256);
        img_2.draw(win_hdc, 332, 32, 256, 256);
        img_3.draw(win_hdc, 632, 32, 256, 256);
    }
}

void updateTrackbars(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    treackbar_squared_grad.update();
    treackbar_learn_rate.update();
    treackbar_momentum.update();

    learn_rate = treackbar_learn_rate.get_pos();
    momentum = treackbar_momentum.get_pos();
    squared_grad = treackbar_squared_grad.get_pos();

    Adam* p = (Adam*)drawer_network.optimizer;
    p->learning_rate = learn_rate;
    p->beta1 = momentum;
    p->beta2 = squared_grad;
}

void onDraw(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    PAINTSTRUCT ps;
    HDC hdc = BeginPaint(hwnd, &ps);

    FillRect(hdc, &ps.rcPaint, (HBRUSH)COLOR_WINDOW);

    img_1.draw(hdc, 32, 32, 256, 256);
    img_2.draw(hdc, 332, 32, 256, 256);
    img_3.draw(hdc, 632, 32, 256, 256);

    EndPaint(hwnd, &ps);
    ReleaseDC(hwnd, hdc);
}

