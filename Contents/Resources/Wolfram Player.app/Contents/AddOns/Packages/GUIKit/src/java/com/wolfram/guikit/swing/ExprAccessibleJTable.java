/*
 * @(#)ExprAccessibleListModel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.util.Vector;

import javax.swing.event.TableModelEvent;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableColumnModel;
import javax.swing.ImageIcon;
import javax.swing.ListSelectionModel;
import javax.swing.table.TableModel;

import com.wolfram.bsf.util.type.ExprTypeConvertor;
import com.wolfram.guikit.type.ExprAccessible;
import com.wolfram.guikit.type.ItemAccessible;
import com.wolfram.guikit.swing.table.JSortTable;
import com.wolfram.guikit.swing.table.SortTableModel;

import com.wolfram.jlink.Expr;

/**
 * ExprAccessibleJTable extends JTable
 */
public class ExprAccessibleJTable extends JSortTable implements ExprAccessible, ItemAccessible {

    private static final long serialVersionUID = -1287987275156781948L;
    
	private static final ExprTypeConvertor convertor = new ExprTypeConvertor();
	
	public ExprAccessibleJTable() {
		this(new ItemTableModel(), null, null);
		}
	public ExprAccessibleJTable(int numRows, int numColumns) {
		this(new ItemTableModel(numRows, numColumns));
		}
	public ExprAccessibleJTable(final Object[][] rowData, final Object[] columnNames) {
    this(new ItemTableModel(rowData, columnNames));
		}
	public ExprAccessibleJTable(Vector rowData, Vector columnNames) {
		this(new ItemTableModel(rowData, columnNames));
		}
	public ExprAccessibleJTable(SortTableModel dm) {
		super(dm);
		}
	public ExprAccessibleJTable(SortTableModel dm, TableColumnModel cm) {
		super(dm, cm);
		}
	public ExprAccessibleJTable(SortTableModel dm, TableColumnModel cm, ListSelectionModel sm) {
		super(dm, cm, sm);
		}
			
  public void tableChanged(TableModelEvent e) {
    if (e != null) {
      TableModel mod = getModel();
      int cols = mod.getColumnCount();
      int rows = mod.getRowCount();
      int begin = e.getFirstRow();
      if (begin < 0) begin = 0;
      int end = e.getLastRow();
      int defaultHeight = getRowHeight();
      for (int i = begin; i <= end; ++i) {
      	if (i >= rows) break;
        int parentRowHeight = getRowHeight(i);
        int newParentRowHeight = defaultHeight;
        for (int j = 0; j < cols; ++j) {
          Object c = mod.getValueAt(i, j);
          if (c != null && c instanceof ImageIcon) {
            if (newParentRowHeight < ((ImageIcon)c).getIconHeight())
              newParentRowHeight = ((ImageIcon)c).getIconHeight();
            }
          }
        if (newParentRowHeight != parentRowHeight) setRowHeight(i, newParentRowHeight);
        }
      }
    
    super.tableChanged(e);
    }
      

  // ExprAccessible interface
  											
	public Expr getExpr() {
	  Object result = convertor.convert(getModel().getClass(), Expr.class, getModel());
		return (result instanceof Expr) ? (Expr)result : null;
		}
		
	public void setExpr(Expr e) {
		Object result = convertor.convert(Expr.class, getModel().getClass(), e);
		if (result != null && result instanceof TableModel)
			setModel((TableModel)result);
		}

	public Expr getPart(int i) {
		Expr result = getExpr();
		if (result != null) return result.part(i);
		return null;
		}
	
	public Expr getPart(int[] ia) {
		Expr result = getExpr();
		if (result != null) return result.part(ia);
		return null;
		}
	
	public void setPart(int i, Expr e) {
		if (getModel() instanceof DefaultTableModel) {
			// TODO can we support this by replacing an entire row or default to first column
			int[] ia = {i,1};
			setPart(ia, e);
			}
		}
		
	public void setPart(int[] ia, Expr e) {
		if (ia.length >= 2 && getModel() instanceof DefaultTableModel) {
			DefaultTableModel model = (DefaultTableModel)getModel();
			Object content = convertor.convertExprAsContent(e);
			int useRow = ia[0];
			int useColm = ia[1];
			if (useRow > 0) --useRow;
			else if (useRow < 0) useRow = model.getRowCount() + useRow;
			if (useColm > 0) --useColm;
			else if (useColm < 0) useColm = model.getColumnCount() + useColm;
			model.setValueAt(content, useRow, useColm);
			}
		}
	
  public Object getSelectedItem() {
    TableModel model = getModel();
    if (model == null) return null;
    
    int selRow = getSelectedRow();
    int selCol = getSelectedColumn();
    if (selRow != -1 && selCol != -1) return model.getValueAt(selRow, selCol);
    else return null;
    }
  
  public Object getSelectedItems() {
    TableModel model = getModel();
    if (model == null) return null;
  
    int[] selRows = getSelectedRows();
    int[] selCols = getSelectedColumns();
    int rows = model.getRowCount();
    int cols = model.getColumnCount();
    
    Object[][] data = null;
    
    if (!getColumnSelectionAllowed() && getRowSelectionAllowed()) {
      if (selRows.length == 0) return new Object[][]{{}};
      data = new Object[selRows.length][cols];
      for (int i = 0; i < selRows.length; ++i) {
        for (int j = 0; j < cols; ++j) {
          data[i][j] = model.getValueAt(selRows[i], j);
          }
        }
      }
    else if (getColumnSelectionAllowed() && !getRowSelectionAllowed()) {
      if (selCols.length == 0) return new Object[][]{{}};
      data = new Object[rows][selCols.length];
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < selCols.length; ++j) {
          data[i][j] = model.getValueAt(i, selCols[j]);
          }
        }
      }
    else if (getColumnSelectionAllowed() && getRowSelectionAllowed()) {
      if (selCols.length == 0 || selRows.length == 0) return new Object[][]{{}};
      data = new Object[selRows.length][selCols.length];
      for (int i = 0; i < selRows.length; ++i) {
        for (int j = 0; j < selCols.length; ++j) {
          data[i][j] = model.getValueAt(selRows[i], selCols[j]);
          }
        }
      }
    else {
      int selRow = getSelectedRow();
      int selCol = getSelectedColumn();
      if (selRow != -1 && selCol != -1) return new Object[][]{{model.getValueAt(selRow, selCol)}};
      else return null;
      }
      
    return data;
    }
  
  // ItemAccessible interface
  
  public Object getItems() {
    if (getModel() != null && getModel() instanceof ItemTableModel) {
      return ((ItemTableModel)getModel()).getItems();
      }
    else if (getModel() != null) {
      TableModel model = getModel();
      if (model instanceof DefaultTableModel) {
        Vector rows = ((DefaultTableModel)model).getDataVector();
        int count = rows.size();
        Object[] rowObjects = new Object[count];
        for (int i = 0; i < count; ++i) {
          rowObjects[i] = ((Vector)rows.elementAt(i)).toArray();
          }
        return rowObjects;
        }
      else {
        int rows = model.getRowCount();
        int cols = model.getColumnCount();
        Object[][] data = new Object[rows][cols];
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < cols; ++j) {
            data[i][j] = model.getValueAt(i, j);
            }
          }
        return data;
        }
      }
     else return null;
    }
  
  public void setItems(Object objs) {
    if (getModel() != null && getModel() instanceof ItemTableModel) {
      ((ItemTableModel)getModel()).setItems(objs);
      }
    else {
      ItemTableModel model = new ItemTableModel();
      model.setItems(objs);
      setModel(model);
      }
    
    tableChanged(new TableModelEvent(getModel(), 
       (getModel().getRowCount() > 0 ? 0 : -1), 
       getModel().getRowCount()-1)
       );
    //resizeAndRepaint();
    }
  
  public void setPrototype(Object r) {
    if (getModel() != null && getModel() instanceof ItemTableModel) {
      ((ItemTableModel)getModel()).setPrototype(r);
      }
    }
  public Object getPrototype() {
    if (getModel() != null && getModel() instanceof ItemTableModel) {
      return ((ItemTableModel)getModel()).getPrototype();
      }
    else return null;
    }
    
  public void setColumnEditable(Object r) {
    if (getModel() != null && getModel() instanceof ItemTableModel) {
      ((ItemTableModel)getModel()).setColumnEditable(r);
      }
    }
  public Object getColumnEditable() {
    if (getModel() != null && getModel() instanceof ItemTableModel) {
      return ((ItemTableModel)getModel()).getColumnEditable();
      }
    else return null;
    }
    
    
  public Object getItem(int index) {
    if (getModel() != null && getModel() instanceof ItemTableModel) {
      return ((ItemTableModel)getModel()).getItem(index);
      }
    else if (getModel() != null) {
      Object o = getItems();
      if (o != null && o.getClass() == Object[].class) {
        return ((Object[])o)[index];
        }
      else return null; 
      }
    else return null;
    }

  public void setItem(int index, Object obj) {
    if (getModel() != null && getModel() instanceof ItemTableModel) {
      ((ItemTableModel)getModel()).setItem(index, obj);
      }
    else if (getModel() != null && getModel() instanceof DefaultTableModel){
      DefaultTableModel model = (DefaultTableModel)getModel();
      Vector rowVector = model.getDataVector();
      int columnCount = getColumnCount();
      if (index < rowVector.size()) {
        if (obj != null) {
          if(obj.getClass() == Object[].class) {
            Object[] thisRow = (Object[])obj;
            if (thisRow.length > columnCount)
              model.setColumnCount(thisRow.length);
            if (obj == null) { 
              rowVector.setElementAt(null, index);
              }
            else {
              Object[] objArr = (Object[])obj;
              Vector v = new Vector(objArr.length);
              for (int i=0; i < objArr.length; i++) {
                v.addElement(objArr[i]);
                }
              rowVector.setElementAt(v, index);
              }
            }
          else {
            if (columnCount == 0) {
              columnCount = 1;
              model.setColumnCount(columnCount);
              }
            Vector rowVec = new Vector(columnCount);
            rowVec.setElementAt(obj, 0);
            rowVector.setElementAt(rowVec, index);
            }
          }
        else rowVector.setElementAt(null, index);
        }
      }
    tableChanged(new TableModelEvent(getModel(), 
       (getModel().getRowCount() > 0 ? 0 : -1), 
       getModel().getRowCount()-1)
       );
    //resizeAndRepaint();
    }
    
  }
