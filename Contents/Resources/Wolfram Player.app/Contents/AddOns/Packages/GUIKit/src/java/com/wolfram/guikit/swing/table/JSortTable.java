/*
 * @(#)JSortTable.java 1.14 03/01/23
 */
package com.wolfram.guikit.swing.table;

import java.awt.Color;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;

import java.util.Hashtable;
import java.util.Vector;

import javax.swing.JTable;
import javax.swing.ListSelectionModel;
import javax.swing.event.TableColumnModelEvent;
import javax.swing.event.TableModelEvent;
import javax.swing.table.JTableHeader;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import javax.swing.table.TableColumnModel;

import com.wolfram.guikit.swing.table.renderers.ColorRenderer;

/**
 * JSortTable
 * 
 * Based on JavaPro article
 * http://www.fawcette.com/javapro/2002_08/magazine/columns/visualcomponents/
 * Created by Claude Duguay
 * Copyright (c) 2002
 */
public class JSortTable extends JTable implements MouseListener, MouseMotionListener{
  
  private static final long serialVersionUID = -1284987974456784948L;
    
  protected int sortedColumnIndex = -1;
  protected int sortedColumnModelIndex = -1;
  
  protected boolean sortedColumnAscending = true;
  protected boolean shouldResort = false;
  protected boolean sortingEnabled = true;
  
  public JSortTable()
  {
    this(new DefaultSortTableModel());
  }
  
  public JSortTable(int rows, int cols)
  {
    this(new DefaultSortTableModel(rows, cols));
  }
  
  public JSortTable(Object[][] data, Object[] names)
  {
    this(new DefaultSortTableModel(data, names));
  }
  
  public JSortTable(Vector data, Vector names)
  {
    this(new DefaultSortTableModel(data, names));
  }
  
  public JSortTable(SortTableModel model)
  {
    super(model);
    initSortHeader();
  }

  public JSortTable(SortTableModel model,
    TableColumnModel colModel)
  {
    super(model, colModel);
    initSortHeader();
  }

  public JSortTable(SortTableModel model,
    TableColumnModel colModel,
    ListSelectionModel selModel)
  {
    super(model, colModel, selModel);
    initSortHeader();
    setDefaultRenderer(Color.class, new ColorRenderer());
  }

  protected void initSortHeader()
  {
    JTableHeader header = getTableHeader();
    header.setDefaultRenderer(new SortHeaderRenderer());
    header.addMouseListener(this);
    header.addMouseMotionListener(this);
  }

  public boolean getSortingEnabled() {return sortingEnabled;}
  public void setSortingEnabled(boolean v) {
    sortingEnabled = v;
    }
  
  public void setColumnSortable(Object r) {
    if (getModel() != null && getModel() instanceof SortTableModel) {
      ((SortTableModel)getModel()).setColumnSortable(r);
      }
    }
  public Object getColumnSortable() {
    if (getModel() != null && getModel() instanceof SortTableModel) {
      return ((SortTableModel)getModel()).getColumnSortable();
      }
    else return null;
    }
    
  public void setSortableByDefault(boolean r) {
    if (getModel() != null && getModel() instanceof SortTableModel) {
      ((SortTableModel)getModel()).setSortableByDefault(r);
      }
    }
  public boolean getSortableByDefault() {
    if (getModel() != null && getModel() instanceof SortTableModel) {
      return ((SortTableModel)getModel()).getSortableByDefault();
      }
    else return false;
    }
    
  public void columnMoved(TableColumnModelEvent e) {
    if (e.getFromIndex() != e.getToIndex() && sortedColumnIndex >= 0) {
      if (e.getFromIndex() == sortedColumnIndex)
        sortedColumnIndex = e.getToIndex();
      else if (e.getToIndex() == sortedColumnIndex) {
        int count = getColumnModel().getColumnCount();
        for (int i = 0; i < count; ++i) {
          if (getColumnModel().getColumn(i).getModelIndex() == sortedColumnModelIndex) {
            sortedColumnIndex = i;
            break;
            }
          }
        }
      }
    super.columnMoved(e);
    }
     
 public int getSortedColumnIndex()
  {
  return sortedColumnIndex;
  }
  
 public int getSortedColumnModelIndex()
  {
  return sortedColumnModelIndex;
  }
  
  public boolean isSortedColumnAscending()
  {
    return sortedColumnAscending;
  }
  
  public void tableChanged(TableModelEvent e) {
    super.tableChanged(e);
    
    boolean fireResort = false;
    
    if ((e == null ||
      (e.getColumn() == TableModelEvent.ALL_COLUMNS ||
       e.getColumn() == sortedColumnModelIndex)) &&
       e.getType() == TableModelEvent.UPDATE) {
      fireResort = true;
      }
      
    if (fireResort) {
      final SortTableModel model = (SortTableModel)getModel();
      if (model != null && sortedColumnModelIndex >=0 ) {
        model.sortColumn(sortedColumnModelIndex, sortedColumnAscending);
        repaint();
        }
      }
 
    }

  // Per cell renderer support
  private boolean usingCellRenderers = false;
  private Hashtable cellRenderers = new Hashtable();
  
  public void setCellRenderer(int row, int col, TableCellRenderer renderer) {
    // store in a hash off of row/col and if the hash is not empty flip 
    // usingCellRenderers flag
    if (renderer != null) {
      cellRenderers.put("" + row + "," + col, renderer);
      usingCellRenderers = true;
      }
    else {
      cellRenderers.remove("" + row + "," + col);
      if (cellRenderers.size() == 0) usingCellRenderers = false;
      }
    }
  
  public TableCellRenderer getCellRenderer(int row, int column) {
    if (!usingCellRenderers) return super.getCellRenderer(row, column);
    Object renderer = cellRenderers.get("" + row + "," + column);
    if (renderer != null) return (TableCellRenderer)renderer;
    else return super.getCellRenderer(row, column);
    }
    
  public void removeCellRenderers() {
    cellRenderers.clear();
    usingCellRenderers = false;
    }
    
  // Per cell tooltip support
  private boolean usingCellToolTips = false;
  private Hashtable cellToolTips = new Hashtable();
  
  public void setToolTipText(int row, int col, String text) {
    // store in a hash off of row/col and if the hash is not empty flip 
    // usingCellToolTips flag
    if (text != null && !text.equals("")) {
      cellToolTips.put("" + row + "," + col, text);
      usingCellToolTips = true;
      }
    else {
      cellToolTips.remove("" + row + "," + col);
      if (cellToolTips.size() == 0) usingCellToolTips = false;
      }
    }
    
  public String getToolTipText(MouseEvent event) {
    if (!usingCellToolTips) return super.getToolTipText(event);
    
    Point point = event.getPoint();
    int row = rowAtPoint(point);
    int column = columnAtPoint(point);
    Object tip = cellToolTips.get("" + row + "," + column);
    if (tip != null) return (String)tip;
    else return super.getToolTipText(event);
    }
   
  public void removeCellToolTips() {
    cellToolTips.clear();
    usingCellToolTips = false;
    }
  
  public void mouseReleased(MouseEvent event)
  {
    if (event.getSource().equals(getTableHeader())) {
      if (!shouldResort) return;
      shouldResort = false;
      
      TableColumnModel colModel = getColumnModel();
      int index = colModel.getColumnIndexAtX(event.getX());
      int modelIndex = colModel.getColumn(index).getModelIndex();
      
      SortTableModel model = (SortTableModel)getModel();
      if (model != null && model.isSortable(modelIndex)) {
        boolean reallySort = true;
        
        TableCellEditor editor = getCellEditor();
        if (editor != null) {
          reallySort = editor.stopCellEditing();
          }
        
        if (!reallySort) return;
        
        // toggle ascension, if already sorted
        if (sortedColumnIndex == index) {
          sortedColumnAscending = !sortedColumnAscending;
          }
        sortedColumnIndex = index;
        sortedColumnModelIndex = modelIndex;
        model.sortColumn(sortedColumnModelIndex, sortedColumnAscending);
        }
      }
  }
  
  public void mousePressed(MouseEvent event) {
    if (event.getSource().equals(getTableHeader()) && sortingEnabled)
      shouldResort = true;
    }
  public void mouseDragged(MouseEvent event) {
    if (event.getSource().equals(getTableHeader()))
      shouldResort = false;
    }
  public void mouseClicked(MouseEvent event) {}
  public void mouseEntered(MouseEvent event) {}
  public void mouseExited(MouseEvent event) {}
  public void mouseMoved(MouseEvent e) {}
    
}

