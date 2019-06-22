/*
 * @(#)ItemTableModel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.util.Vector;

import com.wolfram.bsf.util.MathematicaBSFException;
import com.wolfram.bsf.util.type.MathematicaTypeConvertorRegistry;
import com.wolfram.guikit.type.ItemAccessible;
import com.wolfram.guikit.swing.table.DefaultSortTableModel;

/**
 * ItemTableModel extends DefaultTableModel with a utility
 * "item" indexed get/set property which the default model does not
 * provide
 */
public class ItemTableModel extends DefaultSortTableModel implements ItemAccessible {

  private static final long serialVersionUID = -1287987975956789941L;
    
  private Object[] prototype = null;
  private Object[] columnEditable = null;
  
  public ItemTableModel() {
    super();
    }
  
	public ItemTableModel(int rowCount, int columnCount) {
		super(rowCount, columnCount);
		}
		
	public ItemTableModel(Vector columnNames, int rowCount) {
		super(columnNames, rowCount);
		}
		
	public ItemTableModel(Object[] columnNames, int rowCount) {
		this(convertToVector(columnNames), rowCount);
		}
		
	public ItemTableModel(Vector data, Vector columnNames) {
		super(data, columnNames);
		}
		
	public ItemTableModel(Object[][] data, Object[] columnNames) {
		super(data, columnNames);
		}
			
  public void setPrototype(Object r) {
    prototype = ItemListModel.convertToArray(r);
    }
  public Object getPrototype() {
    return prototype;
    }
    
  public void setColumnEditable(Object r) {
    columnEditable = ItemListModel.convertToArray(r);
    }
  public Object getColumnEditable() {
    return columnEditable;
    }
    
  public boolean isCellEditable(int row, int column) {
    Object columnObject = null;
    if (columnEditable != null) {
      if (columnEditable instanceof Object[]) {
        Object[] columnArray = (Object[])columnEditable;
        if (column < columnArray.length) columnObject = columnArray[column];
        else if (columnArray.length >= 1) columnObject = columnArray[0];
        }
      else columnObject = columnEditable;
      }
    if (columnObject != null) {
      if (columnObject instanceof Boolean) return ((Boolean)columnObject).booleanValue();
      }
    return true;
    }
    
  // Here by default we try and support different renderers
  // for any type instead of just Object.class
  // instead of just using row 0 we should also check
  // for a prototypeRow property
  
  public Class getColumnClass(int c) {
    Object protoObject = null;
    if (prototype != null) {
      if (prototype instanceof Object[]) {
        Object[] protoArray = (Object[])prototype;
        if (c < protoArray.length) protoObject = protoArray[c];
        else if (protoArray.length >= 1) protoObject = protoArray[0];
        }
      else protoObject = prototype;
      }
    if (protoObject == null) {
      try {
        protoObject = getValueAt(0, c);
        }
      catch (ArrayIndexOutOfBoundsException ae) {}
      }
    if (protoObject != null) return protoObject.getClass();
    else return Object.class;
    }
    
  public Object getItems() {
  	Vector rows = getDataVector();
  	int count = rows.size();
  	Object[] rowObjects = new Object[count];
  	for (int i = 0; i < count; ++i) {
			rowObjects[i] = ((Vector)rows.elementAt(i)).toArray();
  		}
    return rowObjects;
    }
  
  public static Object[][] convertToArray(Object objs) {
    Object[][] objArray = null;
    if (objs != null) {
      if (objs.getClass() == Object[][].class)
        objArray = (Object[][])objs;
      else {
        try {
          objArray = (Object[][])MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(
            objs.getClass(), objs, Object[][].class);
          }
        catch (MathematicaBSFException me){
          if (objs.getClass() == Object[].class) {
            Object[] vec = (Object[])objs;
            objArray = new Object[vec.length][];
            for (int i = 0; i < vec.length; ++i) {
              try {
                objArray[i] = (Object[])MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(
                  vec[i].getClass(), vec[i], Object[].class);
                }
              catch (MathematicaBSFException mee) {}
              }
            }
          
          }
        }
      }
    return objArray;
    }
    
  public void setItems(Object objs) {
    Object[][] objArray = convertToArray(objs);
    if (objArray != null) {
      int count = objArray.length;
      setRowCount(count);
      for(int i = 0; i < count; ++i)
        setItem(i, objArray[i]);
      }
    else setRowCount(0);
    fireTableDataChanged();
    }
  
  public Object getItem(int index) {
		Vector rowVector = getDataVector();
		if (index < rowVector.size())
			return ((Vector)getDataVector().elementAt(index)).toArray();
	  else return null;
    }

  public void setItem(int index, Object obj) {
  	Vector rowVector = getDataVector();
  	int columnCount = getColumnCount();
  	if (index < rowVector.size()) {
  		if (obj != null) {
        Object[] objArray = ItemListModel.convertToArray(obj);
  			if(objArray != null) {
  				if (objArray.length > columnCount)
  					setColumnCount(objArray.length);
					rowVector.setElementAt(convertToVector(objArray), index);
  				}
			  else {
			  	if (columnCount == 0) {
						columnCount = 1;
						setColumnCount(columnCount);
			  		}
			  	Vector rowVec = new Vector(columnCount);
			  	rowVec.setElementAt(obj, 0);
			  	rowVector.setElementAt(rowVec, index);
			  	}
  			}
  		else rowVector.setElementAt(null, index);
  		}
    }

  }
