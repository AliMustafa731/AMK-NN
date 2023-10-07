
// enable windows visual theme style
#pragma comment(linker,"\"/manifestdependency:type='win32' name='Microsoft.Windows.Common-Controls' version='6.0.0.0' processorArchitecture='*' publicKeyToken='6595b64144ccf1df' language='*'\"")

#include <Windows.h>
#include <CommCtrl.h>
#include <fstream>
#include <string>
#include <ctime>

#include <process.h>
#include <program.h>
#include <loaders.h>

#include <data/dataset.h>
#include <utils/graphics.h>
#include <utils/random.h>
#include <neural_network.h>

//-----------------------------------------------------
//    Initilize The Program
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

void DrawerNetworkDraw(Tensor<Color> &dest, int w, int h);
void train_network_thread(void *args);
void up_scale_thread(void *args);

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
//    Declarations
//-----------------------------------------------------

// the nueral network structure
NeuralNetwork drawer_network;
MSELoss loss_function;
Adam optimizer;

// data
Tensor<float> netowrk_input, network_label;
DataSet dataset, labels;

Image img_1, img_2, img_3;
Tensor<float> f_img_1;

// other
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
    UP_SCALE_BUTTON,
    LOAD_BAR,
    RANDOM_SELECT_BUTTIN,
};


//-----------------------------------------------------
//    Training & Logic of the program
//-----------------------------------------------------

// thread used to train the neural netwrok
void train_network_thread(void *args)
{
    int w = dataset.shape.w;
    int h = dataset.shape.h;
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

                Tensor<float>& network_out = drawer_network.forward(netowrk_input);
                drawer_network.backward( loss_function.gradient(drawer_network, network_label, size) );

                float _loss = network_label[0] - network_out[0];
                loss += _loss * _loss * 0.5f;

                int idx = x + y * w;
                img_2.img[idx].r = (unsigned char)0;
                img_2.img[idx].g = (unsigned char)(55.0f + network_out[0] * 200.0f);
                img_2.img[idx].b = (unsigned char)(55.0f + network_out[0] * 200.0f);
            }
        }

        optimizer.update(drawer_network.parameters);
        img_2.draw(win_hdc, 332, 32, 256, 256);
        SetWindowText(txt[0], std::to_string(loss).c_str());
    }
}

// thread used to upscale the image
void up_scale_thread(void *args)
{
    DrawerNetworkDraw(img_3.img, 256, 256);
    img_3.draw(win_hdc, 632, 32, 256, 256);

    EnableWindow(GetDlgItem((HWND)args, UP_SCALE_BUTTON), TRUE);
    EnableWindow(GetDlgItem((HWND)args, SAVE_MODEL_BUTTON), TRUE);
    EnableWindow(GetDlgItem((HWND)args, LOAD_MODEL_BUTTON), TRUE);
    EnableWindow(GetDlgItem((HWND)args, TRAIN_BUTTON), TRUE);
}


//---------------------------------------------------------------
//  This function uses the neural netwrok
//  and evaluates it's value (Color) at every
//  (x, y) in range (0 to w, 0 to h).
//
//  (w, h) are the diemensions of the image to draw, we can use any scale
//  we want since the function (Neural Network) is continuous
//  and it's inputs are normalized.
//
//  (x, y) coordinates are normalized in range (0, 10)
//  during training and normal running.
//---------------------------------------------------------------
void DrawerNetworkDraw(Tensor<Color> &dest, int w, int h)
{
    float w_f = (float)w;
    float h_f = (float)h;

    for (int x = 0; x < w; x++)
    {
        for (int y = 0; y < h; y++)
        {
            netowrk_input[0] = (float)x * 10.0f / w_f;
            netowrk_input[1] = (float)y * 10.0f / h_f;
            Tensor<float>& network_out = drawer_network.forward(netowrk_input);

            int idx = x + y * w;
            dest[idx].r = (unsigned char)(55.0f + network_out[0] * 200.0f);
            dest[idx].g = (unsigned char)(55.0f + network_out[0] * 200.0f);
            dest[idx].b = (unsigned char)0;
        }
    }
}


// called once when the program is initializing
void onCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    // initilize the neural network
    drawer_network.init(
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
        }
    );

    optimizer = Adam(0.008f, 0.75f, 0.95f);

    netowrk_input.init(drawer_network.in_shape.size());
    network_label.init(drawer_network.output_layer()->out_size);
    loss_function.init(drawer_network.output_layer()->out_size);

    win_hdc = GetDC(hwnd);

    // load data

    if (load_mnist_images("mnist_digits_images.bin", dataset, 0) == false)
    {
        MessageBox(NULL, "Error : can;t load \"mnist_digits_images.bin\", make sure the executable is in it's main directory", "Opss!", MB_OK);
        PostQuitMessage(0);
    }

    img_1 = Image(dataset.shape.w, dataset.shape.h);
    img_2 = Image(dataset.shape.w, dataset.shape.h);
    img_3 = Image(256, 256);

    f_img_1.init(
        dataset.shape.w, dataset.shape.h, 1,
        dataset[rand32() % dataset.samples_num].data
    );
    embed_one_channel_to_color(
        img_1.img.data, f_img_1.data, dataset.sample_size
    );

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

    TB_learn_rate = TrackBar(hwnd, 75, 450, 300, 30, "0.0", "0.1  Learn Rate", 0.0f, 0.1f);
    TB_momentum = TrackBar(hwnd, 75, 500, 300, 30, "0.0", "1.0  Momentum");
    TB_squared_grad = TrackBar(hwnd, 75, 550, 300, 30, "0.0", "1.0  Squared Grad");

    TB_learn_rate.SetPos( optimizer.learning_rate);
    TB_momentum.SetPos( optimizer.beta1);
    TB_squared_grad.SetPos( optimizer.beta2);

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

// called by windows OS when a button is pressed
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
            std::ofstream file;
            file.open(file_name, std::ios::out | std::ios::binary);

            drawer_network.save(file);
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

            drawer_network.load(file);
            optimizer.load(file);

            file.close();

            TB_learn_rate.SetPos( optimizer.learning_rate);
            TB_momentum.SetPos( optimizer.beta1);
            TB_squared_grad.SetPos( optimizer.beta2);

            DrawerNetworkDraw(img_2.img, 28, 28);
            img_2.draw(win_hdc, 332, 32, 256, 256);
        }
    }

    if (win_id == RANDOM_SELECT_BUTTIN)
    {
        // select a random sample image
        // from the loaded dataset
        int idx = rand32() % dataset.samples_num;
        f_img_1.data = dataset[idx].data;
        f_img_1.s = dataset.shape;

        embed_one_channel_to_color(
            img_1.img.data, f_img_1.data, dataset.sample_size
        );

        img_1.draw(win_hdc, 32, 32, 256, 256);
        img_2.draw(win_hdc, 332, 32, 256, 256);
        img_3.draw(win_hdc, 632, 32, 256, 256);
    }
}

void updateTrackbars(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    TB_squared_grad.update();
    TB_learn_rate.update();
    TB_momentum.update();

    optimizer.learning_rate = TB_learn_rate.GetPos();
    optimizer.beta1 = TB_momentum.GetPos();
    optimizer.beta2 = TB_squared_grad.GetPos();
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

