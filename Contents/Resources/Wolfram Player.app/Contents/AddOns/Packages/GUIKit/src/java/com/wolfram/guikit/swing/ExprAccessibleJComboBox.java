/*
 * @(#)ExprAccessibleListModel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.util.Vector;

import javax.swing.ComboBoxEditor;
import javax.swing.ComboBoxModel;
import javax.swing.JComboBox;
import javax.swing.JTextField;

import com.wolfram.bsf.util.type.ExprTypeConvertor;
import com.wolfram.guikit.type.ExprAccessible;
import com.wolfram.guikit.type.ItemAccessible;

import com.wolfram.jlink.Expr;

/**
 * ExprAccessibleListModel extends DefaultListModel
 */
public class ExprAccessibleJComboBox extends JComboBox implements ExprAccessible, ItemAccessible {

    private static final long serialVersionUID = -1281987975456718912L;
    
	private static final ExprTypeConvertor convertor = new ExprTypeConvertor();
	
	private boolean autocomplete = true;
	private boolean autocompleteCaseSensitive = false;
	
	private AutocompleteAgent autocompleteAdapter = new AutocompleteAgent();
	
	public ExprAccessibleJComboBox() {
		this(new ItemComboBoxModel());
		}
		
	public ExprAccessibleJComboBox(final Object[] listData) {
    this();
    ((ItemComboBoxModel)this.getModel()).setItems(listData);
		}
	
	public ExprAccessibleJComboBox(final Vector listData) {
		this(listData.toArray());
		}
	
	public ExprAccessibleJComboBox(ComboBoxModel dataModel) {
		super(dataModel);
		getEditor().getEditorComponent().addKeyListener(autocompleteAdapter);
		}
	
	public boolean getAutocomplete() {return autocomplete;}
	public void setAutocomplete(boolean v) {
		autocomplete = v;
		}
		
	public boolean getAutocompleteCaseSensitive() {return autocompleteCaseSensitive;}
	public void setAutocompleteCaseSensitive(boolean v) {
		autocompleteCaseSensitive = v;
		}
		
  // ExprAccessible interface 
  
	public Expr getExpr() {
	  Object result = convertor.convert(getModel().getClass(), Expr.class, getModel());
		return (result instanceof Expr) ? (Expr)result : null;
		}
		
	public void setExpr(Expr e) {
		Object result = convertor.convert(Expr.class, getModel().getClass(), e);
		if (result != null && result instanceof ComboBoxModel)
			setModel((ComboBoxModel)result);
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
		if (getModel() instanceof ItemComboBoxModel) {
			ItemComboBoxModel model = (ItemComboBoxModel)getModel();
      Object content = convertor.convertExprAsContent(e);
			model.setItem(i, content);
			}
		}
		
	public void setPart(int[] ia, Expr e) {
		// Lists only support one dimension
		setPart(ia[0], e);
		}
	
  public Object getSelectedItems() {
    return getSelectedObjects();
    }
     
  // ItemAccessible interface 
  public Object getItems() {
    if (getModel() != null && getModel() instanceof ItemComboBoxModel)
      return ((ItemComboBoxModel)getModel()).getItems();
    else if (getModel() != null) {
      int count = getModel().getSize();
      Object[] objs = new Object[count];
      for (int i = 0; i < count; ++i)
        objs[i] = getModel().getElementAt(i);
      return objs;
      }
    return null;
    }
  
  public void setItems(Object objs) {
    if (getModel() != null && getModel() instanceof ItemComboBoxModel)
      ((ItemComboBoxModel)getModel()).setItems(objs);
    else {
      Object[] objArray = ItemListModel.convertToArray(objs);
      ItemComboBoxModel model = new ItemComboBoxModel(objArray);
      setModel(model);
      }
    }
    
  public Object getItem(int index) {
    if (getModel() != null && getModel() instanceof ItemComboBoxModel)
      return ((ItemComboBoxModel)getModel()).getItem(index);
    else if (getModel() != null) {
      return getModel().getElementAt(index);
      }
    return null;
    }

  public void setItem(int index, Object obj) {
    if (getModel() != null && getModel() instanceof ItemComboBoxModel)
      ((ItemComboBoxModel)getModel()).setItem(index, obj);
    else {
      Object o = getItems();
      if (o == null || o.getClass() != Object[].class) return;
      Object[] objs = (Object[])o;
      if (index < objs.length) {
        objs[index] = obj;
        setItems(objs);
        }
      }
    }
    
	public void setEditor(ComboBoxEditor anEditor) {
		super.setEditor(anEditor);
		if (anEditor != null) {
			getEditor().getEditorComponent().addKeyListener(autocompleteAdapter);
			}
		}
	

  class AutocompleteAgent extends KeyAdapter {
	
		private boolean lastMatched = false;
		private int lastPos = -1;
		
		public void keyReleased(KeyEvent e) {
			if (!getAutocomplete()) return;
			char ch = e.getKeyChar();
			if (ch == KeyEvent.CHAR_UNDEFINED || (Character.isISOControl(ch) && ch != '\b'))
				return;
			JTextField editor = (JTextField)getEditor().getEditorComponent();
			String str = editor.getText();
			if (str.length() == 0) return;
			int pos = editor.getCaretPosition();
			if (ch == '\b') {
				int selStart = editor.getSelectionStart();
				int selEnd = editor.getSelectionEnd();
				if (lastMatched && selEnd == str.length() && lastPos == selStart) {
					pos -= 1;
					if (pos < 0) pos = 0;
					}
				}
			String strLower = str.toLowerCase();
			boolean matches = false;
			for (int k=0; k < getItemCount(); k++) {
				String item = getItemAt(k).toString();
			  matches = false;
				if (autocompleteCaseSensitive) matches = item.startsWith(str);
				else matches = (item.toLowerCase()).startsWith(strLower);
				if (matches) {
					editor.setText(item);
					editor.setCaretPosition(item.length());
					editor.moveCaretPosition(pos);
					lastPos = pos;
					break;
					}
				}
			lastMatched = matches;
  		}
  	
  	}
  
  }
