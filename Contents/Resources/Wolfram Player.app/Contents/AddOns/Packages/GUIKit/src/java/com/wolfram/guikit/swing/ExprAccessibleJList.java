/*
 * @(#)ExprAccessibleListModel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.util.Vector;

import javax.swing.DefaultListModel;
import javax.swing.JList;
import javax.swing.ListModel;

import com.wolfram.bsf.util.type.ExprTypeConvertor;
import com.wolfram.guikit.type.ExprAccessible;
import com.wolfram.guikit.type.ItemAccessible;

import com.wolfram.jlink.Expr;

/**
 * ExprAccessibleListModel extends DefaultListModel
 */
public class ExprAccessibleJList extends JList implements ExprAccessible, ItemAccessible {

    private static final long serialVersionUID = -1287587975556788958L;
    
	private static final ExprTypeConvertor convertor = new ExprTypeConvertor();
	
	public ExprAccessibleJList() {
		this(new ItemListModel());
		}
		
	public ExprAccessibleJList(final Object[] listData) {
		this();
    ((ItemListModel)this.getModel()).setItems(listData);
		}
	
	public ExprAccessibleJList(final Vector listData) {
		this(listData.toArray());
		}
	
	public ExprAccessibleJList(ListModel dataModel) {
		super(dataModel);
		}
	
  // ExprAccessible interface
  
	public Expr getExpr() {
	  Object result = convertor.convert(getModel().getClass(), Expr.class, getModel());
		return (result instanceof Expr) ? (Expr)result : null;
		}
		
	public void setExpr(Expr e) {
		Object result = convertor.convert(Expr.class, getModel().getClass(), e);
		if (result != null && result instanceof ListModel)
			setModel((ListModel)result);
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
		if (getModel() instanceof DefaultListModel) {
			DefaultListModel model = (DefaultListModel)getModel();
			int useIndex = i;
			if (useIndex > 0) --useIndex;
			else if (useIndex < 0) useIndex = model.getSize() + useIndex;
			Object content = convertor.convertExprAsContent(e);
			model.setElementAt(content, useIndex);
			}
		}
		
	public void setPart(int[] ia, Expr e) {
		// Lists only support one dimension
		setPart(ia[0], e);
		}
	
  public Object getSelectedItem() {
    return getSelectedValue();
    }
  
  public Object getSelectedItems() {
    return getSelectedValues();
    }
    
  // ItemAccessible interface
  
  public Object getItems() {
    if (getModel() != null && getModel() instanceof ItemListModel) {
      return ((ItemListModel)getModel()).getItems();
      }
    else if (getModel() != null) {
      int count = getModel().getSize();
      Object[] objs = new Object[count];
      for (int i = 0; i < count; ++i)
        objs[i] = getModel().getElementAt(i);
      return objs;
      }
     else return null;
    }
  
  public void setItems(Object objs) {
    if (getModel() != null && getModel() instanceof ItemListModel) {
      ((ItemListModel)getModel()).setItems(objs);
      }
    else {
      ItemListModel model = new ItemListModel();
      model.setItems(objs);
      setModel(model);
      }
    }
  
  public Object getItem(int index) {
    if (getModel() != null && getModel() instanceof ItemListModel) {
      return ((ItemListModel)getModel()).get(index);
      }
    else if (getModel() != null) {
      return getModel().getElementAt(index);
      }
    else return null;
    }

  public void setItem(int index, Object obj) {
    if (getModel() != null && getModel() instanceof ItemListModel) {
      ((ItemListModel)getModel()).setItem(index, obj);
      }
    else if (getModel() != null && getModel() instanceof DefaultListModel){
      DefaultListModel model = (DefaultListModel)getModel();
      model.setElementAt(obj, index);
      }
    }
    
  }
