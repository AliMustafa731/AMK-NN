
#include "MainWindow.h"
#include "Image.h"

// Event table for MainWindow
wxBEGIN_EVENT_TABLE(MainWindow, wxFrame)
    EVT_MENU(wxID_OPEN, MainWindow::OnOpen)
    EVT_MENU(wxID_SAVE, MainWindow::OnSave)
    EVT_MENU(wxID_EXIT, MainWindow::OnExit)
    EVT_MENU(wxID_ABOUT, MainWindow::OnAbout)
wxEND_EVENT_TABLE()

MainWindow::MainWindow(const wxString& title, const wxPoint& pos, const wxSize& size)
: wxFrame(NULL, wxID_ANY, title, pos, size)
{
    CreateMenu();
    CreateControls();
    SetupSizers();
}

void MainWindow::CreateMenu()
{
    // Create a menu bar
    wxMenu *menuFile = new wxMenu;
    menuFile->Append(wxID_OPEN, "&Open Model\tCtrl-O");
    menuFile->Append(wxID_SAVE, "&Save Model\tCtrl-S");
    menuFile->AppendSeparator();
    menuFile->Append(wxID_EXIT);

    wxMenu *menuHelp = new wxMenu;
    menuHelp->Append(wxID_ABOUT);

    wxMenuBar *menuBar = new wxMenuBar;
    menuBar->Append(menuFile, "&File");
    menuBar->Append(menuHelp, "&Help");
    SetMenuBar(menuBar);

    // Create a status bar
    CreateStatusBar();
    SetStatusText("By: Ali Mustafa Kamel, 2022 - 2025");
}

void MainWindow::CreateControls()
{
    m_panel = new wxPanel(this, wxID_ANY);

    m_image_desired = new Image(m_panel, 28, 28, 256, 256);
    m_image_output = new Image(m_panel, 28, 28, 256, 256);

    m_image_output->getBitmapHandle()->SetDoubleBuffered(true);
    m_image_desired->getBitmapHandle()->SetDoubleBuffered(true);

    m_button_train = new wxButton(m_panel, wxID_ANY, "Train", wxDefaultPosition, wxSize(120, 40));
}

void MainWindow::SetupSizers()
{
    wxBoxSizer* mainSizer = new wxBoxSizer(wxHORIZONTAL);

    wxBoxSizer* sizer1 = new wxBoxSizer(wxVERTICAL);
    wxBoxSizer* sizer2 = new wxBoxSizer(wxVERTICAL);

    sizer1->Add(m_image_desired->getBitmapHandle(), wxSizerFlags().Align(wxHORIZONTAL));
    sizer1->AddSpacer(20);
    sizer1->Add(m_button_train, wxSizerFlags().Align(wxHORIZONTAL));
    sizer1->AddSpacer(20);
    sizer1->Add(m_image_desired->getBitmapHandle(), wxSizerFlags().Align(wxHORIZONTAL));

    mainSizer->Add(sizer1, wxSizerFlags().Expand().Proportion(1).Align(wxHORIZONTAL));
    m_panel->SetSizer(mainSizer);
    mainSizer->SetSizeHints(this);
}

// Event handlers
void MainWindow::OnExit(wxCommandEvent& event)
{
    Close(true);
}

void MainWindow::OnAbout(wxCommandEvent& event)
{
    wxMessageBox("By: Ali Mustafa Kamel, 2022 - 2025", "About", wxOK | wxICON_INFORMATION);
}

void MainWindow::OnOpen(wxCommandEvent& event)
{
    
}

void MainWindow::OnSave(wxCommandEvent& event)
{
    
}
