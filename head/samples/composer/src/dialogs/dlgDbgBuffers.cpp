#include "dlgDbgBuffers.h"
#include "../glInfo.h"
#include <nau.h>
#include <nau/debug/profile.h>

BEGIN_EVENT_TABLE(DlgDbgBuffers, wxDialog)
	EVT_GRID_CELL_CHANGING(DlgDbgBuffers::OnGridCellChange)
	//EVT_GRID_CELL_CHANGED(DlgDbgBuffers::OnGridCellChange)

	EVT_PG_SELECTED(DLG_MI_PGBUFFERS, DlgDbgBuffers::OnBufferSelection)
	EVT_PG_CHANGED(DLG_MI_PGBUFFERS, DlgDbgBuffers::OnBufferChanged)
	EVT_SPINCTRL(DLG_MI_GRIDBUFFERINFOLENGTH, DlgDbgBuffers::OnBufferValuesLengthChange)
	EVT_SPINCTRL(DLG_MI_GRIDBUFFERINFOLINES, DlgDbgBuffers::OnBufferValuesLinesChange)
	EVT_SPINCTRL(DLG_MI_GRIDBUFFERINFOPAGE, DlgDbgBuffers::OnBufferValuesPageChange)

	EVT_BUTTON(DLG_MI_SAVEBUFFER, OnSaveBufferInfo)
	EVT_BUTTON(DLG_MI_SAVEVALUEPAGE, OnSavePageInfo)
	EVT_BUTTON(DLG_MI_SAVEVAO, OnSaveVaoInfo)
END_EVENT_TABLE()


wxWindow *DlgDbgBuffers::m_Parent = NULL;
DlgDbgBuffers *DlgDbgBuffers::m_Inst = NULL;
std::map<int, DlgDbgBuffers::BufferSettings> DlgDbgBuffers::bufferSettingsList;
 


void 
DlgDbgBuffers::SetParent(wxWindow *p) {

	m_Parent = p;
}

DlgDbgBuffers* 
DlgDbgBuffers::Instance () {

	if (m_Inst == NULL)
		m_Inst = new DlgDbgBuffers();

	return m_Inst;
}
 

DlgDbgBuffers::DlgDbgBuffers(): wxDialog(DlgDbgBuffers::m_Parent, -1, wxT("Nau - Buffers Information"),wxDefaultPosition,
						   wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
{
	currentBufferIndex = -1;
	this->SetSizeHints( wxDefaultSize, wxDefaultSize);

	wxBoxSizer *bSizer1;
	bSizer1 = new wxBoxSizer( wxVERTICAL);

	wxStaticBoxSizer * sbSizer1;
	sbSizer1 = new wxStaticBoxSizer( new wxStaticBox( this, wxID_ANY, wxEmptyString ), wxVERTICAL );

	wxNotebook *notebook = new wxNotebook(this, -1);
	//wxNotebookSizer *nbSizer = new wxNotebookSizer(notebook);

	/* Buffers */
	wxPanel *pBuffers = new wxPanel(notebook, -1);
	notebook->AddPage(pBuffers, wxT("Buffers"));

	wxSizer *buffers = new wxBoxSizer(wxVERTICAL);

	//Splitter Init
	wxSplitterWindow *buffersSplitter = new wxSplitterWindow(pBuffers, wxID_ANY);
	wxPanel *buffersLeftPanel = new wxPanel(buffersSplitter, wxID_ANY);
	wxPanel *buffersRightPanel = new wxPanel(buffersSplitter, wxID_ANY);

	wxSizer *buffersInformation = new wxBoxSizer(wxVERTICAL); //PropertyGridSizer


	wxSizer *buffersGridInformation = new wxBoxSizer(wxVERTICAL); //Grid Sizer
	wxSizer *buffersGridInformationTop = new wxBoxSizer(wxHORIZONTAL); //Grid Header
	wxSizer *buffersGridInformationBottom = new wxBoxSizer(wxHORIZONTAL); //Grid Footer 


	pBuffers->SetAutoLayout(TRUE);
	pBuffers->SetSizer(buffers);

	//Sizer finishing touches
	buffersLeftPanel->SetSizer(buffersInformation);
	buffersLeftPanel->SetAutoLayout(true);
	buffersRightPanel->SetSizer(buffersGridInformation);
	buffersRightPanel->SetAutoLayout(true);

	buffersSplitter->SplitVertically(buffersLeftPanel, buffersRightPanel);
	buffersSplitter->SetSashPosition(200);
	buffersSplitter->SetMinimumPaneSize(200);
	buffers->Add(buffersSplitter, 1, wxEXPAND, 1);


	//PropertyGrid
	pgBuffers = new wxPropertyGridManager(buffersLeftPanel, DLG_MI_PGBUFFERS,
		wxDefaultPosition, wxDefaultSize,
		wxPG_BOLD_MODIFIED | wxPG_SPLITTER_AUTO_CENTER |
		wxPGMAN_DEFAULT_STYLE
		);

	pgBuffers->AddPage(wxT("Buffers"));

	buffersInformation->Add(pgBuffers, 1, wxALIGN_LEFT | wxGROW | wxALL);

	//Top side of grid

	wxStaticText *lblt1 = new wxStaticText(buffersRightPanel, -1, wxT("Length: "));
	buffersGridInformationTop->Add(lblt1, 0, wxGROW | wxALL, 5);

	spinBufferLength = new wxSpinCtrl(buffersRightPanel, DLG_MI_GRIDBUFFERINFOLENGTH, wxEmptyString, wxDefaultPosition, wxSize(64, -1));
	spinBufferLength->SetRange(1, 4096);
	
	spinBufferLength->SetValue(1);

	buffersGridInformationTop->Add(spinBufferLength, 0, wxALIGN_CENTER | wxALL, 5);


	wxStaticText *lblt2 = new wxStaticText(buffersRightPanel, -1, wxT("Lines: "));
	buffersGridInformationTop->Add(lblt2, 0, wxGROW | wxALL, 5);

	spinBufferLines = new wxSpinCtrl(buffersRightPanel, DLG_MI_GRIDBUFFERINFOLINES, wxEmptyString, wxDefaultPosition, wxSize(64, -1));
	spinBufferLines->SetRange(1, 1024);
	spinBufferLines->SetValue(1);

	buffersGridInformationTop->Add(spinBufferLines, 0, wxALIGN_CENTER | wxALL, 5);

	buffersGridInformation->Add(buffersGridInformationTop, 0, wxALIGN_CENTER | wxALL, 5);


	//grid
	gridBufferValues = new wxGrid(buffersRightPanel, DLG_MI_GRIDBUFFERINFO, wxDefaultPosition);
	gridBufferValues->SetDefaultColSize(70);
	gridBufferValues->SetDefaultRowSize(25);
	gridBufferValues->SetRowLabelSize(40);
	gridBufferValues->SetColLabelSize(0);
	gridBufferValues->CreateGrid(1, 0);
	gridBufferValues->SetRowLabelValue(0, wxString("Type"));
	gridBufferValues->DisableDragColMove();
	gridBufferValues->EnableDragCell(false);
	gridBufferValues->DisableCellEditControl();

	buffersGridInformation->Add(gridBufferValues, 1, wxEXPAND);

	//Bottom side of grid


	wxStaticText *lblb = new wxStaticText(buffersRightPanel, -1, wxT("Page: "));
	buffersGridInformationBottom->Add(lblb, 0, wxGROW | wxALL, 5);

	spinBufferPage = new wxSpinCtrl(buffersRightPanel, DLG_MI_GRIDBUFFERINFOPAGE);
	spinBufferPage->SetRange(1, 1);
	spinBufferPage->SetValue(1);

	buffersGridInformationBottom->Add(spinBufferPage, 1, wxGROW | wxALL);

	buffersGridInformation->Add(buffersGridInformationBottom, 0, wxALIGN_CENTER | wxALL, 5);

	/* VAOs */
	wxPanel *pVAOs = new wxPanel(notebook, -1);
	notebook->AddPage(pVAOs, wxT("VAOs"));

	wxSizer *vaos = new wxBoxSizer(wxVERTICAL);

	//	setupShaders();

	pVAOs->SetAutoLayout(TRUE);
	pVAOs->SetSizer(vaos);

	pgVAOs = new wxPropertyGridManager(pVAOs, DLG_MI_PGVAOS,
		wxDefaultPosition, wxDefaultSize,
		// These and other similar styles are automatically
		// passed to the embedded wxPropertyGrid.
		wxPG_BOLD_MODIFIED | wxPG_SPLITTER_AUTO_CENTER |
		// Plus defaults.
		wxPGMAN_DEFAULT_STYLE
		);

	pgVAOs->AddPage(wxT("VAO"));

	vaos->Add(pgVAOs, 1, wxEXPAND);


	sbSizer1->Add(notebook, 1, wxALL | wxEXPAND, 5);
	bSizer1->Add(sbSizer1, 1, wxEXPAND, 5);
	
	wxBoxSizer *bSizer2;
	bSizer2 = new wxBoxSizer(wxHORIZONTAL);

	m_bSavebuffers = new wxButton(this, DLG_MI_SAVEBUFFER, wxT("Save Buffers"));
	bSizer2->Add(m_bSavebuffers, 0, wxALIGN_CENTER | wxALL, 5);

	m_bSavepage = new wxButton(this, DLG_MI_SAVEVALUEPAGE, wxT("Save Value Page"));
	m_bSavepage->Disable();
	bSizer2->Add(m_bSavepage, 0, wxALIGN_CENTER | wxALL, 5);

	m_bSavevaos = new wxButton(this, DLG_MI_SAVEVAO, wxT("Save VAOs"));
	bSizer2->Add(m_bSavevaos, 0, wxALIGN_CENTER | wxALL, 5);

	bSizer1->Add(bSizer2, 0, wxALIGN_CENTER | wxALL, 5);

	bSizer1->SetSizeHints(this);
	bSizer1->Fit(this);

	this->SetSizer(bSizer1);
	this->Layout();
	this->Centre(wxBOTH);

	clear();
}


void
DlgDbgBuffers::updateDlg() 
{

}


std::string &
DlgDbgBuffers::getName ()
{
	name = "DlgDbgBuffers";
	return(name);
}


DlgDbgBuffers::BufferSettings::BufferSettings() :
length(4),
lines(16),
types({ DLG_FLOAT, DLG_FLOAT, DLG_FLOAT, DLG_FLOAT })
{
}

void
DlgDbgBuffers::eventReceived(const std::string &sender, const std::string &eventType, nau::event_::IEventData *evt)
{
}


void DlgDbgBuffers::append(std::string s) {

}


void DlgDbgBuffers::clear(bool fullclear) {

	isLogClear = true;

	pgBuffers->ClearPage(0);

	pgVAOs->ClearPage(0);

	if (fullclear){
		bufferSettingsList.clear();
	}
}
// Update Dialog
void DlgDbgBuffers::loadBufferInfo() {
	if (isLogClear){
		std::vector<std::pair<std::pair<int, int>, std::vector<int>>> vaoInfoData;

		getCurrentVAOInfoData(vaoInfoData);

		for (std::pair<std::pair<int, int>, std::vector<int>> vaoInfo : vaoInfoData){
			wxPGProperty *pid;
			wxPGProperty *appended, *buffers;

			pid = pgVAOs->Append(new wxPGProperty(wxT("VAO " + std::to_string(vaoInfo.first.first)), wxPG_LABEL));

			appended = pgVAOs->AppendIn(pid, new wxIntProperty(wxT("Element Array"), wxPG_LABEL,
				vaoInfo.first.second));
			appended->Enable(false);

			buffers = pgVAOs->AppendIn(pid, new wxPGProperty(wxT("Buffers"), wxPG_LABEL));
			//pid->Enable(false);
			for (int bufferName : vaoInfo.second){
				appended = pgVAOs->AppendIn(buffers, new wxIntProperty(wxT("Buffer"), wxPG_LABEL,
					bufferName));
				appended->Enable(false);
			}
			pgVAOs->Expand(pid);
		}
		loadBufferInfoGrid();
	}
}




void DlgDbgBuffers::loadBufferInfoGrid(){
	std::map<int, NauGlBufferInfo> *bufferInfoMap = getBufferInfoMap();
	std::map<int, NauGlBufferInfo>::iterator iter;


	for (iter = bufferInfoMap->begin(); iter != bufferInfoMap->end(); iter++){
		NauGlBufferInfo bufferInfo = iter->second;
		wxPGProperty *pid, *appended, *userpg;
		bool newSettings = false;

		pid = pgBuffers->Append(new wxPGProperty(wxT("Buffer " + std::to_string(bufferInfo.index)), wxPG_LABEL));
		pid->SetAttribute(wxT("buffer"), wxVariant(bufferInfo.index));
		if (bufferSettingsList.find(bufferInfo.index) == bufferSettingsList.end()){
			bufferSettingsList[bufferInfo.index] = BufferSettings();
			newSettings = true;
		}
		if (bufferInfo.isVAOBuffer()){
			std::string valueHeader;
			pid->Enable(false);
			appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("Size (Bytes)"), wxPG_LABEL, bufferInfo.size));
			appended->Enable(false);
			appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("Components"), wxPG_LABEL, bufferInfo.components));
			appended->Enable(false);
			appended = pgBuffers->AppendIn(pid, new wxStringProperty(wxT("Type"), wxPG_LABEL, getDatatypeString(bufferInfo.type)));
			appended->Enable(false);
			appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("Stride"), wxPG_LABEL, bufferInfo.stride));
			appended->Enable(false);
			appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("Normalized"), wxPG_LABEL, bufferInfo.normalized));
			appended->Enable(false);
			appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("Divisor"), wxPG_LABEL, bufferInfo.divisor));
			appended->Enable(false);
			appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("Integer"), wxPG_LABEL, bufferInfo.integer));
			appended->Enable(false);

			if (newSettings){
				bufferSettingsList[bufferInfo.index].types.clear();
				bufferSettingsList[bufferInfo.index].types.push_back(getDLGDataType(bufferInfo.type));
				bufferSettingsList[bufferInfo.index].length = bufferInfo.components;
			}
			//appended = pgBuffers->AppendIn(pid, new wxPGProperty(wxT("Values"), wxPG_LABEL));
			//values->Enable(false);
			//valueline = 0;
			//for (int valueNum = 0; valueNum < bufferInfo.second.size(); valueNum++){
			//	if (valueNum % 4 == 0){
			//		if (valueNum + 3 < bufferInfo.second.size()){
			//			valueHeader = "[" + std::to_string(valueNum + 1) + " <-> " + std::to_string(valueNum + 4) + "]";
			//		}
			//		else{
			//			valueHeader = "[" + std::to_string(valueNum + 1) + " <-> " + std::to_string(bufferInfo.second.size()) + "]";
			//		}
			//		valueline = pgBuffers->AppendIn(values, new wxStringProperty(valueHeader, wxPG_LABEL, wxT("<composed>")));
			//		valueline->Enable(false);
			//	}
			//	appended = pgBuffers->AppendIn(valueline, new wxStringProperty(wxT("[" + std::to_string(valueNum + 1) + "]"), wxPG_LABEL, bufferInfo.second[valueNum]));
			//	appended->Enable(false);
			//}
			//pid->Enable(false);
		}
		else{
			appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("Size (Bytes)"), wxPG_LABEL, bufferInfo.size));
			appended->Enable(false);
		}


		int pagesize, linesize = 0;
		int length = bufferSettingsList[bufferInfo.index].length;
		int lines = bufferSettingsList[bufferInfo.index].lines;

		userpg = pgBuffers->AppendIn(pid, new wxPGProperty(wxT("User Settings"), wxPG_LABEL));

		appended = pgBuffers->AppendIn(userpg, new wxIntProperty(wxT("Length"), wxPG_LABEL, length));
		appended = pgBuffers->AppendIn(userpg, new wxIntProperty(wxT("Lines per Page"), wxPG_LABEL, lines));

		//Setting Types Values
		if (bufferInfo.isVAOBuffer()){
			linesize = getBufferDataTypeSize(bufferSettingsList[bufferInfo.index].types[0]) * length;
		}
		else{
			for (int t = 0; t < length; t++){
				linesize += getBufferDataTypeSize(bufferSettingsList[bufferInfo.index].types[t]);
			}
		}

		pagesize = linesize * lines;

		appended = pgBuffers->AppendIn(userpg, new wxIntProperty(wxT("Line Size (bytes)"), wxPG_LABEL, linesize));
		appended->Enable(false);
		appended = pgBuffers->AppendIn(userpg, new wxIntProperty(wxT("Lines"), wxPG_LABEL, (bufferInfo.size / linesize) + 1));
		appended->Enable(false);
		appended = pgBuffers->AppendIn(userpg, new wxIntProperty(wxT("Page Size (bytes)"), wxPG_LABEL, pagesize));
		appended->Enable(false);
		appended = pgBuffers->AppendIn(userpg, new wxIntProperty(wxT("Pages"), wxPG_LABEL, (bufferInfo.size / pagesize) + 1));
		appended->Enable(false);

	}
	pgBuffers->CollapseAll();
}

void DlgDbgBuffers::OnBufferSettingsChange(){
	if (currentBufferIndex > 0){
		int pagesize = 0;
		NauGlBufferInfo bufferInfo;
		const wxString dataType[] = { wxString("BYTE"), wxString("UNSIGNED_BYTE"), wxString("INT"), wxString("UNSIGNED_INT"), wxString("SHORT"), wxString("UNSIGNED_SHORT"), wxString("FLOAT"), wxString("DOUBLE")};
		//const long dataTypeInd[] = { DLG_BYTE, DLG_UNSIGNED_BYTE, DLG_INT, DLG_UNSIGNED_INT, DLG_SHORT, DLG_UNSIGNED_SHORT, DLG_FLOAT, DLG_DOUBLE };

		getBufferInfoFromMap(currentBufferIndex, bufferInfo);

		int length = bufferSettingsList[currentBufferIndex].length;
		int lines = bufferSettingsList[currentBufferIndex].lines;

		m_bSavepage->Enable();

		while (gridBufferValuesHeaders.size() < length){
			gridBufferValuesHeaders.push_back(new wxGridCellChoiceEditor(8, dataType, false));
		}

		//Setting Rows
		if (gridBufferValues->GetNumberRows() > (lines + 1)){
			gridBufferValues->DeleteRows(1, gridBufferValues->GetNumberRows() - (lines + 1));
		}
		else for (int append = gridBufferValues->GetNumberRows(); append < lines + 1; append++){
			gridBufferValues->AppendRows();
			gridBufferValues->SetRowLabelValue(append, wxString(std::to_string(append)));
			for (int col = 1; col < gridBufferValues->GetNumberCols(); col++){
				gridBufferValues->SetReadOnly(append, col);
			}
		}

		//Setting Columns
		if (gridBufferValues->GetNumberCols() > length){
			//!!! WX ERROR UNSUPPORTED!!!
			//while (gridBufferValues->GetNumberCols() > length){
			//	gridBufferValues->DeleteCols();
			//}
			for (int col = length; col < gridBufferValues->GetNumberCols(); col++){
				gridBufferValues->SetReadOnly(0, col);
				for (int row = 0; row < gridBufferValues->GetNumberRows(); row++){
					gridBufferValues->SetCellValue(row, col, wxString(""));
				}
			}
		}
		else for (int append = gridBufferValues->GetNumberCols(); append < length; append++){
			gridBufferValues->AppendCols();
			gridBufferValues->SetCellEditor(0, append, gridBufferValuesHeaders[append]);
			for (int row = 1; row < gridBufferValues->GetNumberRows(); row++){
				gridBufferValues->SetReadOnly(row, append);
			}
		}



		//Setting Types
		if (bufferInfo.isVAOBuffer()){
			for (int col = 0; col < length; col++){
				gridBufferValues->SetReadOnly(0, col);
				gridBufferValues->SetCellValue(0, col, dataType[bufferSettingsList[currentBufferIndex].types[0]]);
			}
		}
		else{
			while (bufferSettingsList[currentBufferIndex].types.size() < length){
				bufferSettingsList[currentBufferIndex].types.push_back(DLG_FLOAT);
			}
			for (int col = 0; col < length; col++){
				gridBufferValues->SetReadOnly(0, col, false);
				gridBufferValues->SetCellValue(0, col, dataType[bufferSettingsList[currentBufferIndex].types[col]]);
			}
		}


		//Setting Types Values
		if (bufferInfo.isVAOBuffer()){
			pagesize = getBufferDataTypeSize(bufferSettingsList[currentBufferIndex].types[0]) * length * lines;
		}
		else{
			for (int t = 0; t < length; t++){
				pagesize += getBufferDataTypeSize(bufferSettingsList[currentBufferIndex].types[t]);
			}
			pagesize *= lines;
		}

		if (bufferInfo.size % pagesize>0){
			spinBufferPage->SetRange(1, (bufferInfo.size / pagesize) + 1);
		}
		else{
			spinBufferPage->SetRange(1, bufferInfo.size / pagesize);
		}
		spinBufferPage->SetValue(1);

		updateBufferData();
	} 
}


void DlgDbgBuffers::updateBufferData(){
	int pagesize = 0, prevBuffer;
	int realSize, page, size;
	int length = bufferSettingsList[currentBufferIndex].length;
	int lines = bufferSettingsList[currentBufferIndex].lines;
	NauGlBufferInfo bufferInfo;		
	getBufferInfoFromMap(currentBufferIndex, bufferInfo);
	std::vector<int> sizes;
	std::vector<void*> pointers;

	page = spinBufferPage->GetValue() - 1;
	realSize = bufferInfo.size;

	for (int t = 0; t < length; t++){
		if (bufferInfo.isVAOBuffer()){
			size = getBufferDataTypeSize(bufferSettingsList[currentBufferIndex].types[0]);
		}
		else{
			size = getBufferDataTypeSize(bufferSettingsList[currentBufferIndex].types[t]);
		}
		pagesize += size;
		sizes.push_back(size);
	}
	pagesize *= lines;

	for (int i = 1; i < lines; i++){
		for (int t = 0; t < length; t++){
			sizes.push_back(sizes[t]);
		}
	}


	prevBuffer = openBufferMapPointers(currentBufferIndex, page, pagesize, realSize, sizes, pointers);
	updateBufferDataValues(pointers);
	//if (realSize < totalsize){
	//	pgBuffers->AppendIn(valuesProperty, new wxStringProperty(wxT("ERROR!"), wxPG_LABEL, wxT("User size exceeds buffer size!")))->Enable(false);
	//}

	closeBufferMapPointers(prevBuffer);
}

void DlgDbgBuffers::updateBufferDataValues(std::vector<void*> &pointers){
	int length = bufferSettingsList[currentBufferIndex].length;
	int lines = bufferSettingsList[currentBufferIndex].lines;
	int pointerIndex;
	NauGlBufferInfo bufferInfo;
	std::string value;

	getBufferInfoFromMap(currentBufferIndex, bufferInfo);

	for (int row = 0; row < lines; row++){
		for (int col = 0; col < length; col++){
			pointerIndex = (length * row) + col;
			if (pointerIndex < pointers.size()){
				if (bufferInfo.isVAOBuffer()){
					value = getStringFromPointer(bufferSettingsList[currentBufferIndex].types[0], pointers[pointerIndex]);
				}
				else{
					value = getStringFromPointer(bufferSettingsList[currentBufferIndex].types[col], pointers[pointerIndex]);
				}
			}
			else{
				value = "";
			}
			gridBufferValues->SetCellValue(row + 1, col, value);
		}
	}

}

DlgDbgBuffers::DataTypes DlgDbgBuffers::getBufferDataType(wxPGProperty *typeProperty, int index){
	wxVariant type = typeProperty->GetPropertyByName("Type " + std::to_string(index))->GetValue();
	return static_cast<DataTypes>(type.GetLong());
}

DlgDbgBuffers::DataTypes DlgDbgBuffers::getDLGDataType(int type){
	switch (type){
	case GL_UNSIGNED_BYTE:
		return DLG_UNSIGNED_BYTE;
	case GL_BYTE:
		return DLG_BYTE;
	case GL_UNSIGNED_SHORT:
		return DLG_UNSIGNED_SHORT;
	case GL_SHORT:
		return DLG_SHORT;
	case GL_UNSIGNED_INT:
		return DLG_UNSIGNED_INT;
	case GL_INT:
		return DLG_INT;
	//case GL_HALF_FLOAT:
	//	return DLG_HALF_FLOAT;
	case GL_FLOAT:
		return DLG_FLOAT;
	}
	return DLG_DOUBLE;
}

DlgDbgBuffers::DataTypes DlgDbgBuffers::getDLGDataType(std::string type){
	map<std::string, DataTypes> m = { 
			{ "UNSIGNED_BYTE", DLG_UNSIGNED_BYTE },
			{ "BYTE", DLG_BYTE },
			{ "UNSIGNED_SHORT", DLG_UNSIGNED_SHORT },
			{ "SHORT", DLG_SHORT },
			{ "UNSIGNED_INT", DLG_UNSIGNED_INT },
			{ "INT", DLG_INT },
			{ "FLOAT", DLG_FLOAT },
			{ "DOUBLE", DLG_DOUBLE } };
	return m[type];
}

int DlgDbgBuffers::getBufferDataTypeSize(DataTypes type){
	switch (type){
	case DLG_BYTE:
	case DLG_UNSIGNED_BYTE:
		return sizeof(char);
	case DLG_INT:
	case DLG_UNSIGNED_INT:
		return sizeof(int);
	case DLG_SHORT:
	case DLG_UNSIGNED_SHORT:
		return sizeof(short);
	case DLG_FLOAT:
		return sizeof(float);
	case DLG_DOUBLE:
		return sizeof(double);
	}
	return 0;
}



std::string DlgDbgBuffers::getStringFromPointer(DataTypes type, void* ptr){
	int intconverter;
	unsigned int uintconverter;
	//try{
		switch (type){
		case DLG_BYTE:
			intconverter = *((char*)ptr);
			return std::to_string(intconverter);
		case DLG_UNSIGNED_BYTE:
			uintconverter = *((unsigned char*)ptr);
			return std::to_string(uintconverter);
		case DLG_INT:
			intconverter = *((int*)ptr);
			return std::to_string(intconverter);
		case DLG_UNSIGNED_INT:
			uintconverter = *((unsigned int*)ptr);
			return std::to_string(uintconverter);
		case DLG_SHORT:
			return std::to_string(*((short*)ptr));
		case DLG_UNSIGNED_SHORT:
			return std::to_string(*((unsigned short*)ptr));
		case DLG_FLOAT:
			return std::to_string(*((float*)ptr));
		case DLG_DOUBLE:
			return std::to_string(*((double*)ptr));
		}
	//}
	//catch(){
	//	return "ERROR!";
	//}
	return "unknown";
}


void DlgDbgBuffers::OnBufferSelection(wxPropertyGridEvent& e){
	wxPGProperty *bufferProperty = e.GetProperty();

	if (bufferProperty){
		while (!bufferProperty->GetLabel().StartsWith("Buffer")){
			bufferProperty = bufferProperty->GetParent();
		}
		currentBufferIndex = bufferProperty->GetAttribute(wxT("buffer")).GetLong();
		loadBufferSettings();


	}
}

void DlgDbgBuffers::loadBufferSettings(){
	if (currentBufferIndex >= 0){
		spinBufferLength->SetValue(bufferSettingsList[currentBufferIndex].length);
		spinBufferLines->SetValue(bufferSettingsList[currentBufferIndex].lines);
		spinBufferPage->SetValue(1);
		OnBufferSettingsChange();
	}
}


void DlgDbgBuffers::OnBufferValuesLengthChange(wxSpinEvent& e){
	if (currentBufferIndex >= 0){
		int newValue = spinBufferLength->GetValue();
		if (bufferSettingsList[currentBufferIndex].length != newValue){
			bufferSettingsList[currentBufferIndex].length = newValue;
			wxPGProperty *settings = pgBuffers->GetCurrentPage()->GetRoot()->
				GetPropertyByName("Buffer " + std::to_string(currentBufferIndex))->GetPropertyByName("User Settings");
			settings->GetPropertyByName("Length")->SetValue(newValue);
			OnBufferSettingsChange();
			loadBufferSettingsPGUpdate(settings, currentBufferIndex);
		}
	}
}
void DlgDbgBuffers::OnBufferValuesLinesChange(wxSpinEvent& e){
	if (currentBufferIndex >= 0){
		int newValue = spinBufferLines->GetValue();
		if (bufferSettingsList[currentBufferIndex].lines != newValue){
			bufferSettingsList[currentBufferIndex].lines = newValue;
			wxPGProperty *settings = pgBuffers->GetCurrentPage()->GetRoot()->
				GetPropertyByName("Buffer " + std::to_string(currentBufferIndex))->GetPropertyByName("User Settings");
			settings->GetPropertyByName("Lines per Page")->SetValue(newValue);
			OnBufferSettingsChange();
			loadBufferSettingsPGUpdate(settings, currentBufferIndex);
		}
	}
}
void DlgDbgBuffers::OnBufferValuesPageChange(wxSpinEvent& e){
	if (currentBufferIndex >= 0){
		updateBufferData();
	}
}

void DlgDbgBuffers::OnGridCellChange(wxGridEvent& event){
	if (event.GetRow() == 0){
		if (currentBufferIndex >= 0){
			//wxString typeString = gridBufferValues->GetCellValue(event.GetRow(), event.GetCol());
			wxString typeString = event.GetString();
			DataTypes type = getDLGDataType(typeString.ToStdString());
			if (bufferSettingsList[currentBufferIndex].types[event.GetCol()] != type){
				bufferSettingsList[currentBufferIndex].types[event.GetCol()] = type;
				OnBufferSettingsChange();
				wxPGProperty *settings = pgBuffers->GetCurrentPage()->GetRoot()->
					GetPropertyByName("Buffer " + std::to_string(currentBufferIndex))->GetPropertyByName("User Settings");
				settings->GetPropertyByName("Lines per Page");
				loadBufferSettingsPGUpdate(settings, currentBufferIndex);
			}
		}
	}
}

void DlgDbgBuffers::OnSaveBufferInfo(wxCommandEvent& event){
	wxFileDialog dialog(this,
		wxT("Buffer info save file dialog"),
		wxEmptyString,
		wxT("bufferinfo.txt"),
		wxT("Text files (*.txt)|*.txt"),
		wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
	if (wxID_OK == dialog.ShowModal()) {

		wxString path = dialog.GetPath();

		fstream s;
		s.open(path.mb_str(), fstream::out);
		OnSavePropertyGridAux(s, pgBuffers->GetCurrentPage());
		s.close();
	}
}
void DlgDbgBuffers::OnSavePageInfo(wxCommandEvent& event){
	if (currentBufferIndex >= 0){
		wxFileDialog dialog(this,
			wxT("Buffer Values save file dialog"),
			wxEmptyString,
			wxT("bufferpageinfo.txt"),
			wxT("Text files (*.txt)|*.txt"),
			wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
		if (wxID_OK == dialog.ShowModal()) {
			std::string value;
			wxString path = dialog.GetPath();
			int page = spinBufferPage->GetValue();
			bool fail = false;

			fstream s;
			s.open(path.mb_str(), fstream::out);

			s << "Buffer: " << currentBufferIndex <<
				"\tPage: " << page <<
				"\tSize: " << bufferSettingsList[currentBufferIndex].lines << "x" << bufferSettingsList[currentBufferIndex].length << "\n";
			for (int row = 0; row < gridBufferValues->GetNumberRows(); row++){
				for (int col = 0; col < bufferSettingsList[currentBufferIndex].length; col++){
					value = gridBufferValues->GetCellValue(row, col).ToStdString();
					if (value.size() <= 0){
						fail = true;
						break;
					}

					if (col != 0){
						s << "\t";
					}
					s << value;
				}
				if (fail){
					break;
				}
				s << "\n";
			}

			s.close();
		}
	}
}

void DlgDbgBuffers::OnSaveVaoInfo(wxCommandEvent& event){
	wxFileDialog dialog(this,
		wxT("VAO info save file dialog"),
		wxEmptyString,
		wxT("VAOinfo.txt"),
		wxT("Text files (*.txt)|*.txt"),
		wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
	if (wxID_OK == dialog.ShowModal()) {

		wxString path = dialog.GetPath();

		fstream s;
		s.open(path.mb_str(), fstream::out);
		OnSavePropertyGridAux(s, pgVAOs->GetCurrentPage());
		s.close();
	}
}

void DlgDbgBuffers::OnSavePropertyGridAux(std::fstream &s, wxPropertyGridPage *page){
	wxPGProperty *bufferPG, *levelAux;
	wxPropertyGridIterator iterator = page->GetIterator();
	int level;
	bool write;
	while (!iterator.AtEnd()){
		bufferPG = iterator.GetProperty();
		write = true;
		level = 0;
		levelAux = bufferPG;
		while (levelAux->GetParent() != levelAux->GetMainParent()){
			if (levelAux->GetBaseName().Matches("User Settings")){
				write = false;
				break;
			}
			levelAux = levelAux->GetParent();
			level++;
		}
		
		if (write){
			if (level == 0){
				s << bufferPG->GetBaseName().ToStdString() << "\n";
			}
			else{
				for (int i = 0; i < level; i++){
					s << "\t";
				}
				s << bufferPG->GetBaseName().ToStdString() << ":\t" << bufferPG->GetValue().GetString().ToStdString() << "\n";
			}
		}
		iterator.Next();
	}
}

//Valid only for
//appended = pgBuffers->AppendIn(userpg, new wxIntProperty(wxT("Length"), wxPG_LABEL, length));
//appended = pgBuffers->AppendIn(userpg, new wxIntProperty(wxT("Lines per Page"), wxPG_LABEL, lines));
void DlgDbgBuffers::OnBufferChanged(wxPropertyGridEvent& e){
	wxPGProperty *p = e.GetProperty();

	if (p->GetBaseName().Matches("Length")){
		long buffer = p->GetParent()->GetParent()->GetAttribute(wxT("buffer")).GetLong();
		int newValue = p->GetValue().GetLong();

		if (bufferSettingsList[buffer].length != newValue){
			bufferSettingsList[buffer].length = newValue;
			if (buffer == currentBufferIndex){
				spinBufferLength->SetValue(bufferSettingsList[buffer].length);
			}
			loadBufferSettingsPGUpdate(p->GetParent(), buffer);
			OnBufferSettingsChange();
		}

	}
	if (p->GetBaseName().Matches("Lines per Page")){
		long buffer = p->GetParent()->GetParent()->GetAttribute(wxT("buffer")).GetLong();
		int newValue = p->GetValue().GetLong();

		if (bufferSettingsList[buffer].lines != newValue){
			bufferSettingsList[buffer].lines = newValue;
			if (buffer == currentBufferIndex){
				spinBufferLines->SetValue(bufferSettingsList[buffer].lines);
			}
			loadBufferSettingsPGUpdate(p->GetParent(), buffer);
			OnBufferSettingsChange();
		}

	}
}
void DlgDbgBuffers::loadBufferSettingsPGUpdate(wxPGProperty *settings, int buffer){
	NauGlBufferInfo bufferInfo;
	int pagesize, linesize = 0;
	int length = bufferSettingsList[buffer].length;
	int lines = bufferSettingsList[buffer].lines;

	getBufferInfoFromMap(buffer, bufferInfo);

	//Setting Types Values
	if (bufferInfo.isVAOBuffer()){
		linesize = getBufferDataTypeSize(bufferSettingsList[buffer].types[0]) * length;
	}
	else{
		for (int t = 0; t < length; t++){
			linesize += getBufferDataTypeSize(bufferSettingsList[buffer].types[t]);
		}
	}

	pagesize = linesize * lines;

	settings->GetPropertyByName("Line Size (bytes)")->SetValue(linesize);
	settings->GetPropertyByName("Lines")->SetValue((bufferInfo.size / linesize) + 1);
	settings->GetPropertyByName("Page Size (bytes)")->SetValue(pagesize);
	settings->GetPropertyByName("Pages")->SetValue((bufferInfo.size / pagesize) + 1);
}
