#ifndef __IMAGE_GRID_CELL_RENDERER__
#define __IMAGE_GRID_CELL_RENDERER__




class ImageGridCellRenderer : public wxGridCellStringRenderer
{
public:

	wxBitmap *bm;

	ImageGridCellRenderer() { bm = NULL; }

	ImageGridCellRenderer(wxBitmap *bitmap) {
		bm = bitmap;
	}

    virtual void Draw(wxGrid& grid,
                      wxGridCellAttr& attr,
                      wxDC& dc,
                      const wxRect& rect,
                      int row, int col,
                      bool isSelected) {

			wxGridCellStringRenderer::Draw(grid, attr, dc, rect, row, col, isSelected);
			if (bm != NULL) {
				dc.SetBrush(*wxLIGHT_GREY_BRUSH);
				dc.DrawBitmap(*bm,rect.GetLeft()+4,rect.GetTop()+4);
			}
		}

	void setBitmap(wxBitmap *bitmap) {

		bm = bitmap;
	}
};

#endif