#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import xlrd

__all__ = ['read_excel_raw','excel_api']


def read_excel_raw(filedir,sheets=0,skiprows=0,skipcols=0):
    "read excel for multiple sheets, return raw data in list form"
    # input processing, change all inputs to list
    sheets = sheets if isinstance(sheets,(list,tuple)) else [sheets]
    length = len(sheets)
    skiprows = skiprows if isinstance(skiprows,(list,tuple)) else [skiprows]*length
    skipcols = skipcols if isinstance(skipcols,(list,tuple)) else [skipcols]*length
    
    results = []
    with xlrd.open_workbook(filedir, on_demand=True) as wb:
        for j,sheet in enumerate(sheets):
            if isinstance(sheet,int):
                ws = wb.sheet_by_index(sheet)
            elif isinstance(sheet,str):
                ws = wb.sheet_by_name(sheet)
            
            rowskip,colskip = skiprows[j],skipcols[j]
            if colskip:
                data = [ws.row_values(i)[colskip:] for i in range(rowskip,ws.nrows)]
            else:
                data = [ws.row_values(i) for i in range(rowskip,ws.nrows)]
            
            results.append(data)
            
        wb.release_resources()
    return results if length > 1 else data


class win32com_excel(object):
    '''
    Open Excel Application and write data through win32 COM
    Note: args row/col are 1-indexing follows Excel rule
    '''
    def __init__(self, filedir=None, mode='r'):
        
        self.filedir = filedir
        self.mode = mode
        self._initialize_handles()
        
    def _initialize_handles(self):
        # process mode, modify path
        assert self.mode in ('r','w','rw','wr'), \
            "Unexpected mode, accept read('r'), write('w'), readwrite('rw')."
        if isinstance(self.filedir,str) and '\\' not in self.filedir: 
            self.filedir = self.filedir.replace('/','\\')
        
        self.excel = win32com.client.Dispatch("Excel.Application")
        if 'r' in self.mode:
            if self.filedir is None: raise ValueError("Excel directory not provided!")
            self.wb = self.excel.Workbooks.Open(self.filedir)
        else:
            self.excel.Visible = False
            self.excel.DisplayAlerts = False
            self.wb = self.excel.Workbooks.Add()
            if self.filedir is not None: self.wb.SaveAs(self.filedir)
        self.Sheets = self.wb.Sheets
        
    def get_data_shape(self,data,structed=True):
        if type(data).__name__ == 'ndarray':
            shape = data.shape
            ndim = data.ndim
            data = data.tolist()
        elif not data:
            return (None,0,())
        elif not isinstance(data,str) and hasattr(data,'__len__'):
            shape0 = len(data)
            length = lambda e: \
                len(e) if hasattr(e,'__len__') and not isinstance(e,str) else 0
            # try if data has 2nd dim
            elementLen = [length(e) for e in data]; maxLen = max(elementLen)
            ndim, shape = (2,(shape0,maxLen)) if maxLen else (1,(shape0,))
            if not structed: return (data, ndim, shape,elementLen)
        else:
            raise TypeError(f"Unexpected data type {type(data)}!")
        return (data, ndim, shape)
        
    def _process_write_args(self,row,col,data,assertndim):
        # verify row&col
        for n,s in ((row,'row'),(col,'col')):
            assert isinstance(n,int) and n>0, f"arg {s} should be positive!"
        # check mode
        assert 'w' in self.mode, "write access denied for mode "+self.mode
        data, ndim, shape = self.get_data_shape(data)
        # check dimension
        # assert ndim == assertndim, f"data expected to be {assertndim}-dimentional!"
        return (data, shape)
        
    def get_Range(self, sheet, row, col, shape):
        return sheet.Range(sheet.Cells(row,col),
                           sheet.Cells(row+shape[0]-1,col+shape[1]-1))
                           
    def _write(self, sheet, row, col, data, shape):
        self.get_Range(sheet, row, col, shape).Value = data
        
    def write_column(self, sheet, row, col, data):
        data, shape = self._process_write_args(row,col,data,assertndim=1)
        self._write(sheet,row,col,data=list(zip(list(data))),shape=(shape[0],1))
    
    def write_row(self, sheet, row, col, data):
        data, shape = self._process_write_args(row,col,data,assertndim=1)
        self._write(sheet,row,col,data,shape=(1,shape[0]))

    def write_matrix(self, sheet, row, col, data, axis=None):
        data, shape = self._process_write_args(row,col,data,assertndim=2)
        if axis is None: # directly write whole 2Dmatrix
            self._write(sheet, row, col, data, shape)
        elif axis == 0: # write data row by row
            _ = [self._write(sheet,row+i,col,list(rowdata),(1,shape[1])) 
                 for i,rowdata in enumerate(data) if rowdata is not None]
        elif axis == 1: # write data col by col
            _ = [self._write(sheet,row,col+i,list(zip(list(coldata))),(shape[1],1)) 
                 for i,coldata in enumerate(data) if coldata is not None]
        else:
            raise ValueError("Unexpected value of arg 'axis', should be\n" +\
                             "\tNone: directly write whole 2Dmatrix to spreadsheet;\n" +\
                             "\t0:    write data row by row in spreadsheet\n" +\
                             "\t1:    write data column by column in spreadsheet")
            
    def add_sheet(self,sheet_name,data,header=None,index=None,axis=None):
        assert 'w' in self.mode, "write access denied for mode "+self.mode
        # add sheet and sheet name
        [sheet.Delete() for sheet in self.wb.Sheets if sheet.Name == sheet_name]
        sheet = self.Sheets.Add(None,self.Sheets(self.Sheets.count))
        sheet.Name = sheet_name
        row,col = 1,1 # default data start row/col at Cell(1,1)
        # write header
        _, ndim, shape, eleLen = self.get_data_shape(header,structed=False)
        if ndim==1:
            self._write(sheet,row=row,col=col,data=header,shape=(1,shape[0]))
            row += 1
        elif ndim==2:
            _ = [self._write(sheet,row+i,col,rowheader,(1,eleLen[i])) 
                 for i,rowheader in enumerate(header)]
            row += shape[0]
        # write index
        if index:
            self.write_column(sheet,row=row,col=col,data=index)
            col += 1
        # write data to sheet
        self.write_matrix(sheet,row=row,col=col,data=data,axis=axis)
        return sheet
        
    def displayExcelApp(self):
        # delete all irrelavant sheets
        [sheet.Delete() for sheet in self.Sheets if sheet.name[:5]=="Sheet"]
        # save workbook if filedir exists
        if 'w' in self.mode and self.filedir: self.wb.Save()
        self.excel.DisplayAlerts = True
        self.excel.Visible = True
        # self.wb.Close(False)
        # self.excel.Quit()
        
        
class xlsxwriter_excel(object):
    '''
    Write data to excel file through xlsxwriter
    Open Excel Application and based on platform
    Note: args row/col are 1-indexing follows Excel rule
    '''
    def __init__(self, filedir=None, mode='r'):
        self.filedir = filedir
        self.mode = mode
        self._initialize_handles()
        
    def _find_directory(self):
        "find suitable directory if self.filedir is None and write mode"
        available = []; n = 0
        existed = glob.glob('BrowseData*.xlsx')
        # for existed file satisfy pattern, detect if they are opened
        for file in existed:
            try:
                with open(file,"r+") as f: pass
                available.append(file)
            except PermissionError:
                n = max(n,int(file[len('BrowseData'):-len('.xlsx')]))
        # choose filedir and delete unused files
        if available:
            self.filedir = available[0]
            _ = [os.remove(file) for file in available[1:]]
        else:
            self.filedir = f"BrowseData{n+1}.xlsx"
            
    def _initialize_handles(self):
        # process mode, modify path
        assert self.mode in ('r','w','rw','wr'), \
            "Unexpected mode, accept read('r'), write('w'), readwrite('rw')."
        
        if self.filedir is None:
            if 'r' in self.mode:
                raise ValueError("Excel directory not provided!")
            else:
                # find suitable filedir and remove all not opened existed file
                self._find_directory()
        elif '\\' not in self.filedir:
            self.filedir = self.filedir.replace('/','\\')
            
        self.wb = xlsxwriter.Workbook(self.filedir)
        
    def get_data_shape(self,data):
        if not isinstance(data,str) and hasattr(data,'__len__'):
            shape0 = len(data)
            length = lambda e: \
                len(e) if hasattr(e,'__len__') and not isinstance(e,str) else 0
            # try if data has 2nd dim
            maxLen = max([length(e) for e in data])
            ndim, shape = (2,(shape0,maxLen)) if maxLen else (1,(shape0,))
        else:
            raise TypeError(f"Unexpected data type {type(data)}!")
        return (data, ndim, shape)
    
    def write_column(self, sheet, row, col, data):
        assert 'w' in self.mode, "write access denied for mode "+self.mode
        sheet.write_column(row-1, col-1, data)
        
    def write_row(self, sheet, row, col, data):
        assert 'w' in self.mode, "write access denied for mode "+self.mode
        sheet.write_row(row-1, col-1, data)
    
    def write_matrix(self, sheet, row, col, data, axis=1):
        assert 'w' in self.mode, "write access denied for mode "+self.mode
        if axis is None: # directly write whole 2Dmatrix
            raise TypeError("write whole 2Dmatrix unavailable for mac OS system")
        elif axis == 0: # write data row by row
            for i,rowdata in enumerate(data):
                if rowdata is not None:
                    sheet.write_row(row+i-1,col-1,rowdata)
        elif axis == 1: # write data col by col
            for i,coldata in enumerate(data):
                if coldata is not None:
                    sheet.write_column(row-1,col+i-1,coldata)
        else:
            raise ValueError("Unexpected value of arg 'axis', should be\n" +\
                             "\tNone: directly write whole 2Dmatrix to spreadsheet;\n" +\
                             "\t0:    write data row by row in spreadsheet\n" +\
                             "\t1:    write data column by column in spreadsheet")
            
    def add_sheet(self,sheet_name,data,header=None,index=None,axis=0):
        assert 'w' in self.mode, "write access denied for mode "+self.mode
        # add sheet and sheet name
        sheet = self.wb.add_worksheet(sheet_name)
        row,col = 1,1 # default data start row/col at Cell(1,1)
        # write header, allow 2D list
        if header:
            shape0 = len(header)
            has2d = hasattr(header[0],'__len__') and not isinstance(header[0],str)
            if has2d:
                self.write_matrix(sheet, row, col, header, axis=0)
                row += shape0
            else:
                self.write_row(sheet, row, col, header)
                row += 1
        # write index
        if index:
            sheet.write_column(row-1,col-1,index)
            col += 1
        # write data to sheet
        self.write_matrix(sheet,row=row,col=col,data=data,axis=axis)
        return sheet
        
    def displayExcelApp(self):
        "launch excel application based on Platform"
        self.wb.close()
        if sys.platform.lower().startswith('win'):
            os.system(f"start excel {self.filedir}")
        elif sys.platform.lower().startswith('darwin'):
            os.system(f"open -a 'Microsoft Excel.app' '{self.filedir}'")
            
        
# cross-platform handling
if sys.platform.lower().startswith('win'):
    # platform = "Windows"
    import win32com.client
    excel_api = win32com_excel
elif sys.platform.lower().startswith('darwin'):
    # platform = "Mac OS"
    import xlsxwriter,os,glob
    excel_api = xlsxwriter_excel
else:
    raise ImportError(f"Platform {sys.platform} is not supported!")
