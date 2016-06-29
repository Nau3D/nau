#include "dlgBuffers.h"

//#ifdef GLINTERCEPTDEBUG
//#include "..\..\GLIntercept\Src\MainLib\ConfigDataExport.h"
//#endif


#include <nau.h>
#include <nau/system/file.h>

using namespace nau::system;

#include <glbinding/gl/gl.h>
using namespace gl;
//#include <GL/glew.h>

BEGIN_EVENT_TABLE(DlgDbgBuffers, wxDialog)
	EVT_GRID_CELL_CHANGED(DlgDbgBuffers::OnGridCellChange)
	EVT_SPINCTRL(DLG_MI_GRIDBUFFERINFOLENGTH, DlgDbgBuffers::OnBufferValuesLengthChange)
	EVT_SPINCTRL(DLG_MI_GRIDBUFFERINFOLINES, DlgDbgBuffers::OnBufferValuesLinesChange)
	EVT_PG_SELECTED(DLG_MI_PGBUFFERS, DlgDbgBuffers::OnBufferSelection)
	EVT_SPINCTRL(DLG_MI_GRIDBUFFERINFOPAGE, DlgDbgBuffers::OnBufferValuesPageChange)

	EVT_BUTTON(DLG_MI_REFRESH, OnRefreshBufferInfo)
	EVT_BUTTON(DLG_MI_REFRESH_BUFFER_DATA, OnRefreshBufferData)
	EVT_BUTTON(DLG_MI_UPDATE_BUFFER, OnUpdateBuffer)

END_EVENT_TABLE()


wxWindow *DlgDbgBuffers::Parent = NULL;
DlgDbgBuffers *DlgDbgBuffers::Inst = NULL;
 

static const std::map<Enums::DataType, std::string> aux = {
	{ Enums::BYTE, std::string("BYTE") },
	{ Enums::UBYTE,  std::string("UBYTE") },
	{ Enums::SHORT,  std::string("SHORT") },
	{ Enums::USHORT, std::string("USHORT") },
	{ Enums::INT,    std::string("INT") },
	{ Enums::UINT,   std::string("UINT") },
	{ Enums::FLOAT,  std::string("FLOAT") },
	{ Enums::DOUBLE, std::string("DOUBLE") } };
std::map<Enums::DataType, std::string> DlgDbgBuffers::DataType = aux;


void 
DlgDbgBuffers::SetParent(wxWindow *p) {

	Parent = p;
}


DlgDbgBuffers* 
DlgDbgBuffers::Instance () {

	if (Inst == NULL)
		Inst = new DlgDbgBuffers();

	return Inst;
}
 

DlgDbgBuffers::DlgDbgBuffers(): wxDialog(DlgDbgBuffers::Parent, -1, wxT("Buffer Info"),wxDefaultPosition,
						   wxDefaultSize,wxRESIZE_BORDER|wxDEFAULT_DIALOG_STYLE)
{
	currentBuffer = NO_BUFFER;
	m_UseShortNames = true;

	std::vector<wxString> vs;
	for (auto s : DataType) {

		vs.push_back(s.second);
	}
	for (int i = 0; i < MAX_COLUMNS; ++i) {

		gridBufferValuesHeaders.push_back(new wxGridCellChoiceEditor(8, (wxString *)&vs[0], false));
	}
	
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

	wxStaticText *lblt1 = new wxStaticText(buffersRightPanel, -1, wxT("Columns: "));
	buffersGridInformationTop->Add(lblt1, 0, wxGROW | wxALL, 5);

	spinBufferLength = new wxSpinCtrl(buffersRightPanel, DLG_MI_GRIDBUFFERINFOLENGTH, wxEmptyString, wxDefaultPosition, wxSize(64, -1));
	spinBufferLength->SetRange(1, MAX_COLUMNS);
	
	spinBufferLength->SetValue(1);

	buffersGridInformationTop->Add(spinBufferLength, 0, wxALIGN_CENTER | wxALL, 5);


	wxStaticText *lblt2 = new wxStaticText(buffersRightPanel, -1, wxT("Lines: "));
	buffersGridInformationTop->Add(lblt2, 0, wxGROW | wxALL, 5);

	spinBufferLines = new wxSpinCtrl(buffersRightPanel, DLG_MI_GRIDBUFFERINFOLINES, wxEmptyString, wxDefaultPosition, wxSize(64, -1));
	spinBufferLines->SetRange(1, MAX_ROWS);
	spinBufferLines->SetValue(1);

	buffersGridInformationTop->Add(spinBufferLines, 0, wxALIGN_CENTER | wxALL, 5);

	wxStaticText *lblb = new wxStaticText(buffersRightPanel, -1, wxT("Page: "));
	buffersGridInformationTop->Add(lblb, 0, wxGROW | wxALL, 5);

	spinBufferPage = new wxSpinCtrl(buffersRightPanel, DLG_MI_GRIDBUFFERINFOPAGE, wxEmptyString, wxDefaultPosition, wxSize(128, -1));
	spinBufferPage->SetRange(1, 1);
	spinBufferPage->SetValue(1);

	buffersGridInformationTop->Add(spinBufferPage, 0, wxALIGN_CENTER | wxALL, 5);

	buffersGridInformation->Add(buffersGridInformationTop, 0, wxALIGN_CENTER | wxALL, 5);


	//grid
	gridBufferValues = new wxGrid(buffersRightPanel, DLG_MI_GRIDBUFFERINFO, wxDefaultPosition);
	gridBufferValues->SetDefaultColSize(70);
	gridBufferValues->SetDefaultRowSize(25);
	gridBufferValues->SetRowLabelSize(40);
	gridBufferValues->SetColLabelSize(0);
	gridBufferValues->CreateGrid(1, MAX_COLUMNS);
	gridBufferValues->SetRowLabelValue(0, wxString("Type"));
	gridBufferValues->DisableDragColMove();
	gridBufferValues->EnableDragCell(false);
	gridBufferValues->DisableCellEditControl();
	for (int col = 0; col < MAX_COLUMNS; ++col) {

		gridBufferValues->SetCellEditor(0, col, gridBufferValuesHeaders[col]);
		gridBufferValues->SetCellValue(0, col, wxT("FLOAT"));
		gridBufferValues->HideCol(col);
	}


	buffersGridInformation->Add(gridBufferValues, 1, wxEXPAND);

	//Bottom side of grid

	/* VAOs */
	wxPanel *pVAOs = new wxPanel(notebook, -1);
	notebook->AddPage(pVAOs, wxT("VAOs"));

	wxSizer *vaos = new wxBoxSizer(wxVERTICAL);

	//	setupBuffers();

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

	m_bRefresh = new wxButton(this, DLG_MI_REFRESH, wxT("Refresh Buffers"));
	bSizer2->Add(m_bRefresh, 0, wxALIGN_CENTER | wxALL, 5);

	m_bRefreshBufferData = new wxButton(this, DLG_MI_REFRESH_BUFFER_DATA, wxT("Refresh Buffer Data"));
	bSizer2->Add(m_bRefreshBufferData, 0, wxALIGN_CENTER | wxALL, 5);

	m_bUpdateBuffer = new wxButton(this, DLG_MI_UPDATE_BUFFER, wxT("Update Buffer"));
	bSizer2->Add(m_bUpdateBuffer, 0, wxALIGN_CENTER | wxALL, 5);

	//	m_bSavevaos = new wxButton(this, DLG_MI_SAVEVAO, wxT("Save VAOs"));
//	bSizer2->Add(m_bSavevaos, 0, wxALIGN_CENTER | wxALL, 5);

	bSizer1->Add(bSizer2, 0, wxALIGN_CENTER | wxALL, 5);

	bSizer1->SetSizeHints(this);
	bSizer1->Fit(this);

	this->SetSizer(bSizer1);
	this->Layout();
	this->Centre(wxBOTH);

	clear();
}


void
DlgDbgBuffers::updateDlg(bool shortNames) {

	clear();

	m_UseShortNames = shortNames;
	setVAOList();
	setBufferList();
	setBufferData();
}


// ((VAO index, (Element index, Element name)) , vector(Array index, Array Name))
//std::vector<std::pair<std::pair<int, std::pair<int, std::string>>, std::vector<std::pair<int, std::string>>>> VAOList;


void
DlgDbgBuffers::setVAOList(void) {

	VAOInfoList list;
	// ((VAO index, (Element index, Element name)) , vector(Array index, Array Name))
	VAOInfo vao;
	int id, count, enabled;
	std::string label;

	pgVAOs->ClearPage(0);
//#ifdef GLINTERCEPTDEBUG
//	gliSetIsGLIActive(false);
//#endif
	// ugly but practical :-)
	for (int i = 0; i < 65536; ++i) {

		if ((boolean)glIsVertexArray(i)) {

			vao.first.first = i;
			glBindVertexArray(i);

			// get element array buffer name
			glGetIntegerv(GL_ELEMENT_ARRAY_BUFFER_BINDING, &id);
			if (id) {

				vao.first.second.first = id;
				label = RESOURCEMANAGER->getBufferByID(id)->getLabel();
				if (m_UseShortNames)
					name = File::GetName(label);
				else
					name = label;
				vao.first.second.second = name;
			}
			else {
				vao.first.second.first = 0;
			}

			vao.second.clear();
			// get info for each attrib mapped buffer
			glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &count);
			for (int k = 0; k < count; ++k) {

				glGetVertexAttribiv(k, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled);
				if (enabled) {
					glGetVertexAttribiv(k, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &id);
					IBuffer *b = RESOURCEMANAGER->getBufferByID(id);
					if (b) {
						label = b->getLabel();
						if (m_UseShortNames)
							name = File::GetName(label);
						else
							name = label;

						vao.second.push_back(std::pair<int, std::string>(id, name));
					}
				}
			}
			list.push_back(vao);
		}
	}
//#ifdef GLINTERCEPTDEBUG
//	gliSetIsGLIActive(true);
//#endif

	for (auto info : list) {

		wxPGProperty *pid;
		wxPGProperty *element, *buffers;

		pid = pgVAOs->Append(new wxPGProperty(wxT("VAO " + std::to_string(info.first.first)), wxPG_LABEL));

		// Add index buffer
		if (info.first.second.first != 0) {

			element = pgVAOs->AppendIn(pid, new wxPGProperty(wxT("Index Array"), wxPG_LABEL));
			pgVAOs->AppendIn(element,
				new wxStringProperty(wxT("ID " + std::to_string(info.first.second.first)),
				wxPG_LABEL,
				wxT(" " + info.first.second.second)));
		}

		// Add Vertex Attributes
		buffers = pgVAOs->AppendIn(pid, new wxPGProperty(wxT("Attribute Arrays"), wxPG_LABEL));
		for (auto buf : info.second) {

			pgVAOs->AppendIn(buffers,
				new wxStringProperty(wxT("ID " + std::to_string(buf.first)),
				wxPG_LABEL,
				wxT(" " + buf.second)));

		}
	}
	pgVAOs->CollapseAll();
}


void DlgDbgBuffers::setBufferList(){

	std::vector<std::string> names;
	IBuffer *b;
	wxPGProperty *pid, *appended;

	std::string indexName, shortName, fullName;

	pgBuffers->ClearPage(0);
	bufferSettingsList.clear();
	currentBuffer = NO_BUFFER;

	RESOURCEMANAGER->getBufferNames(&names);

	for (auto name : names) {

		fullName = name;
		shortName = File::GetName(name);
		if (m_UseShortNames)
			indexName = shortName;
		else
			indexName = fullName;

		b = RESOURCEMANAGER->getBuffer(name);
		b->refreshBufferParameters();
		std::vector<Enums::DataType> structure = b->getStructure();

		bufferSettingsList[indexName] = BufferSettings();
		if (structure.size() == 0)
			bufferSettingsList[indexName].types.push_back(Enums::FLOAT);
		else
			bufferSettingsList[indexName].types = b->getStructure();
		bufferSettingsList[indexName].lines = 16;
		bufferSettingsList[indexName].currentPage = 0;
		bufferSettingsList[indexName].ID = b->getPropi(IBuffer::ID);
		bufferSettingsList[indexName].size = b->getPropui(IBuffer::SIZE);
		bufferSettingsList[indexName].bufferPtr = b;
		bufferSettingsList[indexName].fullName = fullName;
		bufferSettingsList[indexName].shortName = shortName;

		int s = 0;
		for (auto t : bufferSettingsList[indexName].types) {
			s += Enums::getSize(t);
		}
		bufferSettingsList[indexName].lineSize = s;

		int totalLines = bufferSettingsList[indexName].size / s;
		if (bufferSettingsList[indexName].size % s != 0)
			totalLines++;

		int totalPages = totalLines / (bufferSettingsList[indexName].lines);
		if (totalLines % bufferSettingsList[indexName].lines != 0)
			totalPages++;

		pid = pgBuffers->Append(new wxStringProperty(wxT("" + indexName), wxPG_LABEL,"<composed>"));
		appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("ID"), wxPG_LABEL, bufferSettingsList[indexName].ID));
		appended->Enable(false);
		appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("Size (bytes)"), wxPG_LABEL, bufferSettingsList[indexName].size));
		appended->Enable(false);
		appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("Total pages"), wxPG_LABEL, totalPages));
		appended->Enable(false);
		appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("Total lines"), wxPG_LABEL, totalLines));
		appended->Enable(false);
		appended = pgBuffers->AppendIn(pid, new wxIntProperty(wxT("Components"), wxPG_LABEL, bufferSettingsList[indexName].types.size()));
		appended->Enable(false);
	}

	if (bufferSettingsList.size() > 0)
		currentBuffer = m_UseShortNames ? File::GetName(names[0]):indexName;

	pgBuffers->CollapseAll();
}


void
DlgDbgBuffers::setBufferData(){

	gridBufferValues->ClearGrid();

	if (currentBuffer == NO_BUFFER) {
		return;
	}

	int lines = bufferSettingsList[currentBuffer].lines;
	int page = bufferSettingsList[currentBuffer].currentPage;
	int lineSize = bufferSettingsList[currentBuffer].lineSize;
	int columns = bufferSettingsList[currentBuffer].types.size();

	void *bufferValues;
	int pageSize = lineSize * lines;
	int pageOffset = page * pageSize;
	bufferValues = malloc(pageSize);

	IBuffer *b = bufferSettingsList[currentBuffer].bufferPtr;
//#ifdef GLINTERCEPTDEBUG
//	gliSetIsGLIActive(false);
//#endif

	int dataRead = b->getData(pageOffset, pageSize, bufferValues);

//#ifdef GLINTERCEPTDEBUG
//	gliSetIsGLIActive(true);
//#endif

	setSpinners(lines, columns, page, lineSize, bufferSettingsList[currentBuffer].size);


	int pointerIndex;
	std::string value;

	char rowLabel[32];

	int k = gridBufferValues->GetNumberRows();
	if (k < lines+1)
		gridBufferValues->AppendRows(lines+1-k);
	if (k > lines + 1)
		gridBufferValues->DeleteRows(lines + 1, k - lines - 1);
	k = gridBufferValues->GetNumberCols();
	if ( k < columns)
		gridBufferValues->AppendCols(columns-k);

	pointerIndex = 0;

	for (int col = 0; col < columns; ++col) {

		Enums::DataType t = bufferSettingsList[currentBuffer].types[col];
		std::string s = DataType[t];
		gridBufferValues->SetCellValue(0, col, wxT(""+s));
		gridBufferValues->ShowCol(col);
	}
	for (int col = columns; col < MAX_COLUMNS; ++col) {
	
		gridBufferValues->HideCol(col);
	}

	for (int row = 0; row < lines; ++row){

		sprintf(rowLabel, "%d", row + page*lines);
		gridBufferValues->SetRowLabelValue(row + 1, wxString(rowLabel));

		

		for (int col = 0; col < columns; ++col){
			if (pointerIndex < dataRead){
				void *ptr = (char *)bufferValues + pointerIndex;
				value = Enums::pointerToString(bufferSettingsList[currentBuffer].types[col], ptr);
			}
			else{
				value = "";
			}
			gridBufferValues->SetCellValue(row + 1, col, value);
			pointerIndex += Enums::getSize(bufferSettingsList[currentBuffer].types[col]);
		}
	}
	free(bufferValues);
}


void 
DlgDbgBuffers::setSpinners(int lines, int columns, int page, int lineSize, int size) {


	int totalLines = size / lineSize;
	if (size % lineSize != 0)
		totalLines++;
	int totalPages = totalLines / lines;
	if (totalLines % lines != 0)
		totalPages++;

	spinBufferLines->SetRange(1, totalLines);
	spinBufferPage->SetRange(1, totalPages);
	spinBufferLines->SetValue(lines);
	spinBufferLength->SetValue(columns);
	spinBufferPage->SetValue(page + 1);

	if (page > 0) {
		spinBufferLines->Disable();
		spinBufferLength->Disable();
	}
	else {
		spinBufferLines->Enable();
		spinBufferLength->Enable();
	}
}


void
DlgDbgBuffers::setBufferProperties() {

	int lineSize = bufferSettingsList[currentBuffer].lineSize;
	int totalLines = (int)ceilf(bufferSettingsList[currentBuffer].size * 1.0 / lineSize);
	int totalPages = (int)ceilf(totalLines * 1.0 / bufferSettingsList[currentBuffer].lines);
	int components = bufferSettingsList[currentBuffer].types.size();


	wxPGProperty *pid = pgBuffers->GetProperty(wxT(""+currentBuffer));

	wxPGProperty *child = pid->GetPropertyByName(wxT("Total lines"));
	child->SetValue(totalLines);
	child = pid->GetPropertyByName(wxT("Total pages"));
	child->SetValue(totalPages);
	child = pid->GetPropertyByName(wxT("Components"));
	child->SetValue(components);
	pgBuffers->Refresh();
}


void
DlgDbgBuffers::OnUpdateBuffer(wxCommandEvent& event) {

	int elemSize;
	std::string elem;
	int lines = bufferSettingsList[currentBuffer].lines;
	int columns = bufferSettingsList[currentBuffer].types.size();
	int size = bufferSettingsList[currentBuffer].size;
	int lineSize = bufferSettingsList[currentBuffer].lineSize;
	int page = bufferSettingsList[currentBuffer].currentPage;

	void *buffer = malloc(lineSize * lines);
	char *bufferPtr = (char *)buffer;
	int offSet = page * lineSize * lines;
	int dataRead = 0;

	for (int row = 1; row <= lines; ++row){

		for (int col = 0; col < columns; ++col){

			if (offSet + dataRead < size){
				elem = gridBufferValues->GetCellValue(row, col);
				insertIntoBuffer(elem, bufferSettingsList[currentBuffer].types[col], bufferPtr);
				elemSize = Enums::getSize(bufferSettingsList[currentBuffer].types[col]);
				dataRead += elemSize;
				bufferPtr += elemSize;
			}
		}
	}
//#ifdef GLINTERCEPTDEBUG
//	gliSetIsGLIActive(false);
//#endif

	bufferSettingsList[currentBuffer].bufferPtr->setSubData(offSet, dataRead, buffer);
//#ifdef GLINTERCEPTDEBUG
//	gliSetIsGLIActive(true);
//#endif
}


void 
DlgDbgBuffers::insertIntoBuffer(std::string elem, Enums::DataType type, void *ptr) {

	int size;
	unsigned int valueui;
	int valuei;
	char valueb;
	byte valueub;
	short values;
	unsigned short valueus;

	float valuef;
	double valued;

	size = Enums::getSize(type);
	switch (type) {

		case Enums::UBYTE:
			valueub = (byte)strtoul(elem.c_str(), NULL, 0);
			memcpy(ptr, &valueub, size);
			break;
		case Enums::BYTE:
			valueb = (char)strtoul(elem.c_str(), NULL, 0);
			memcpy(ptr, &valueb, size);
			break;
		case Enums::USHORT:
			valueus = (unsigned short)strtoul(elem.c_str(), NULL, 0);
			memcpy(ptr, &valueus, size);
			break;
		case Enums::SHORT:
			values = (short)strtoul(elem.c_str(), NULL, 0);
			memcpy(ptr, &values, size);
			break;
		case Enums::UINT:
			valueui = strtoul(elem.c_str(), NULL, 0);
			memcpy(ptr, &valueui, size);
			break;
		case Enums::INT:
			valuei = (int)strtoul(elem.c_str(), NULL, 0);
			memcpy(ptr, &valuei, size);
			break;
		case Enums::FLOAT:
			valuef = strtof(elem.c_str(), NULL);
			memcpy(ptr, &valuef, size);
			break;
		case Enums::DOUBLE:
			valued = strtod(elem.c_str(), NULL);
			memcpy(ptr, &valued, size);
			break;
	}
}



void 
DlgDbgBuffers::OnRefreshBufferData(wxCommandEvent& event) {

	setBufferData();
}


void
DlgDbgBuffers::OnRefreshBufferInfo(wxCommandEvent& event) {

	updateDlg();
}


void 
DlgDbgBuffers::OnGridCellChange(wxGridEvent& event) {

	if (currentBuffer == NO_BUFFER)
		return;

	if (event.GetRow() == 0){
		wxString typeString = gridBufferValues->GetCellValue(event.GetRow(), event.GetCol());
		//wxString typeString = event.GetString();
		Enums::DataType t = Enums::getType(typeString.ToStdString());
		if (bufferSettingsList[currentBuffer].types[event.GetCol()] != t) {
			bufferSettingsList[currentBuffer].types[event.GetCol()] = t;
			setBufferData();
			gridBufferValues->Refresh();
		}
	}
}


void DlgDbgBuffers::OnBufferValuesLengthChange(wxSpinEvent& e) {

	if (currentBuffer == NO_BUFFER)
		return;

	int newValue = spinBufferLength->GetValue();
	int prev = bufferSettingsList[currentBuffer].types.size();
	if (prev != newValue) {
		Enums::DataType lastColumn = bufferSettingsList[currentBuffer].types[prev - 1];

		// if smaller
		while (prev > newValue) {
			bufferSettingsList[currentBuffer].types.pop_back();
			--prev;
		}
		// if greater
		while (prev < newValue) {
			bufferSettingsList[currentBuffer].types.push_back(lastColumn);
			++prev;
		}

		int ls = 0;
		for (unsigned int i = 0; i < bufferSettingsList[currentBuffer].types.size(); ++i)
			ls += Enums::getSize(bufferSettingsList[currentBuffer].types[i]);

		bufferSettingsList[currentBuffer].lineSize = ls;
		
		setBufferProperties();
		setBufferData();
	}
}


void 
DlgDbgBuffers::OnBufferValuesLinesChange(wxSpinEvent& e) {

	if (currentBuffer == NO_BUFFER)
		return;

	int newValue = spinBufferLines->GetValue();
	
	bufferSettingsList[currentBuffer].lines = newValue;
	setBufferProperties();
	setBufferData();
}


void
DlgDbgBuffers::OnBufferValuesPageChange(wxSpinEvent& e) {

	if (currentBuffer == NO_BUFFER)
		return;

	int newValue = spinBufferPage->GetValue();

	bufferSettingsList[currentBuffer].currentPage = newValue-1;
	setBufferData();
}


void 
DlgDbgBuffers::OnBufferSelection(wxPropertyGridEvent& e) {

	const wxPGProperty *p = e.GetProperty();
	if (p == NULL)
		return;
	const wxPGProperty *q = p->GetParent();
	if (q->GetLabel() == "<Root>")
		currentBuffer = p->GetLabel();
	else
		currentBuffer = q->GetLabel();
	setBufferData();
}



std::string &
DlgDbgBuffers::getName () {

	name = "DlgDbgBuffers";
	return(name);
}


DlgDbgBuffers::BufferSettings::BufferSettings() :
	lines(16),
	currentPage(0),
	ID(0),
	size(0),
	types({}),
	lineSize(0)
{
}

void
DlgDbgBuffers::eventReceived(const std::string &sender, const std::string &eventType, 
	const std::shared_ptr<IEventData> &evt) {

}


void DlgDbgBuffers::clear() {

	pgBuffers->ClearPage(0);
	pgVAOs->ClearPage(0);
	currentBuffer = NO_BUFFER;
	gridBufferValues->ClearGrid();
}


std::string 
DlgDbgBuffers::getStringFromPointer(Enums::DataType type, void* ptr){
	int intconverter;
	unsigned int uintconverter;
	//try{
		switch (type){
			case Enums::BYTE:
				intconverter = *((char*)ptr);
				return std::to_string(intconverter);
			case Enums::UBYTE:
				uintconverter = *((unsigned char*)ptr);
				return std::to_string(uintconverter);
			case Enums::INT :
				intconverter = *((int*)ptr);
				return std::to_string(intconverter);
			case Enums::UINT:
				uintconverter = *((unsigned int*)ptr);
				return std::to_string(uintconverter);
			case Enums::SHORT:
				return std::to_string(*((short*)ptr));
			case Enums::USHORT:
				return std::to_string(*((unsigned short*)ptr));
			case Enums::FLOAT:
				return std::to_string(*((float*)ptr));
			case Enums::DOUBLE:
				return std::to_string(*((double*)ptr));
		}
	//}
	//catch(){
	//	return "ERROR!";
	//}
	return "unknown";
}

/*
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
		spinBufferPage->SetValue(bufferSettingsList[currentBufferIndex].currentPage+1);
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
		int newValue = spinBufferPage->GetValue();
		bufferSettingsList[currentBufferIndex].currentPage = newValue-1;
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

*/