
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
#include <painter.h>

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

PainterNetwork painter_network;
Adam optimizer;

float learn_rate = 0.008f;
float momentum = 0.75f;
float squared_grad = 0.95f;
bool training = false;

Image img_1, img_2, img_3;

const char* header_txt = "\nBy : Ali Mustafa Kamel\n2022-2023";

OPENFILENAME of_load_amknn = { 0 };
OPENFILENAME of_save_amknn = { 0 };
OPENFILENAME of_load_image = { 0 };
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

void train_network_thread(void *args)
{
    img_1.draw(win_hdc, 32, 32, 256, 256);
}

void onCreate(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    painter_network = PainterNetwork(Shape(64, 64, 1));
    painter_network.add(new DrawLineElement());
    Tensor<float> trash;
    painter_network.forward(trash);

    img_1 = Image(64, 64);

    denormalize(painter_network.map);
    embed_one_channel_to_color(img_1.img.data, painter_network.map.data, painter_network.map.size());

    img_1.draw(win_hdc, 32, 32, 256, 256);

	win_hdc = GetDC(hwnd);

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
	CreateOPENFILENAME
	(
		&of_load_image, hwnd, file_name, sizeof(file_name), ".png format\0*.png\0\0", "png",
		OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY
	);
	CreateOPENFILENAME
	(
		&of_save_image, hwnd, file_name, sizeof(file_name), ".png format\0*.png\0\0", "png",
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

		}
	}

	if (win_id == LOAD_MODEL_BUTTON)
	{
		if (GetOpenFileName(&of_load_amknn))
		{

		}
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

	optimizer.learning_rate = learn_rate;
    optimizer.beta1 = momentum;
    optimizer.beta2 = squared_grad;
}

void onDraw(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	PAINTSTRUCT ps;
	HDC hdc = BeginPaint(hwnd, &ps);

	FillRect(hdc, &ps.rcPaint, (HBRUSH)COLOR_WINDOW);

	img_1.draw(hdc, 32, 32, 256, 256);
	//img_2.draw(hdc, 332, 32, 256, 256);
	//img_3.draw(hdc, 632, 32, 256, 256);

	EndPaint(hwnd, &ps);
	ReleaseDC(hwnd, hdc);
}

