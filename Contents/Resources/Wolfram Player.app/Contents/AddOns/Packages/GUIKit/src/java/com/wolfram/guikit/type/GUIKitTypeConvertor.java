/*
 * @(#)GUIKitTypeConvertor.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.type;

import com.wolfram.bsf.util.MathematicaBSFException;
import com.wolfram.bsf.util.type.MathematicaTypeConvertor;
import com.wolfram.bsf.util.type.MathematicaTypeConvertorRegistry;

import com.wolfram.guikit.swing.ItemListModel;
import com.wolfram.guikit.swing.ItemTableModel;

import com.wolfram.jlink.Expr;
import com.wolfram.jlink.LoopbackLink;
import com.wolfram.jlink.MathLink;
import com.wolfram.jlink.MathLinkException;
import com.wolfram.jlink.MathLinkFactory;

import java.math.BigDecimal;
import java.math.BigInteger;

import javax.swing.ListModel;
import javax.swing.table.TableModel;
import javax.swing.tree.TreeModel;
import javax.swing.tree.TreeNode;
import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.DefaultMutableTreeNode;

/**
 * GUIKitTypeConvertor
 */
public class GUIKitTypeConvertor implements MathematicaTypeConvertor {

	public Object convert(Class from, Class to, Object obj) {

		if (to == Expr.class) {
			if (from == Expr.class) {
				return obj;
				}
	
			else if (ListModel.class.isAssignableFrom(from)) {
				
				Expr e = null;
				ListModel l = (ListModel)obj;
				LoopbackLink loop = null;
				try {
					loop = MathLinkFactory.createLoopbackLink();
					loop.putFunction("List", l.getSize());
					for (int i=0; i < l.getSize(); ++i) {
						putListContent(loop, l.getElementAt(i));
						}
					e = loop.getExpr();
					}
				catch (MathLinkException me) {}
				finally {
					if (loop != null)
						loop.close();
					}
				return e;
				
				}
				
			else if (TreeModel.class.isAssignableFrom(from)) {
				
				Expr e = null;
				TreeModel t = (TreeModel)obj;
				LoopbackLink loop = null;
				try {
					loop = MathLinkFactory.createLoopbackLink();
					putTreeNode(loop, t, t.getRoot());
					e = loop.getExpr();
					}
				catch (MathLinkException me) {}
				finally {
					if (loop != null)
						loop.close();
					}
				return e;
				
				}
	
			else if (TreeNode.class.isAssignableFrom(from)) {
				
				Expr e = null;
				TreeNode t = (TreeNode)obj;
				LoopbackLink loop = null;
				try {
					loop = MathLinkFactory.createLoopbackLink();
					putTreeNode(loop, t);
					e = loop.getExpr();
					}
				catch (MathLinkException me) {}
				finally {
					if (loop != null)
						loop.close();
					}
				return e;
				
				}
							
			else if (TableModel.class.isAssignableFrom(from)) {
				
				Expr e = null;
				TableModel t = (TableModel)obj;
				LoopbackLink loop = null;
				try {
					loop = MathLinkFactory.createLoopbackLink();
					int rows = t.getRowCount();
					int cols = t.getColumnCount();
					loop.putFunction("List", rows);
					for (int i = 0; i < rows; ++i) {
						loop.putFunction("List", cols);
						for (int j = 0; j < cols; ++j) {
							putTableCellContent(loop, t.getValueAt(i,j));
							}
						}
					e = loop.getExpr();
					}
				catch (MathLinkException me) {}
				finally {
					if (loop != null)
						loop.close();
					}
				return e;
				
				}
				
	    else {
	      return null;
	      }
    	}
		else if (from == Expr.class) {
			try {
			if (to == Expr.class) {
				return obj;
				}
        
		  // TODO sometime we can check if the requested class is
		  // immutable or not and return an instance that matches 
		  // this access
			else if (ListModel.class.isAssignableFrom(to)) {
				ItemListModel m = new ItemListModel();
   
				if (obj == null) return m;
				
				Expr e = (Expr)obj;
				if (e.listQ()) {
					int count = e.length();
					m.setSize(count);
					for (int i = 0; i < count; ++i) {
						m.setElementAt( convertExprAsContent( e.part(i+1)), i);
						}
					}
				
				return m;
				}
				
			// TODO sometime we can check if the requested class is
			// immutable or not and return an instance that matches 
			// this access
			else if (TableModel.class.isAssignableFrom(to)) {
				
				ItemTableModel m = new ItemTableModel();
				
				if (obj == null) return m;
			
				Expr e = (Expr)obj;
				if (e.matrixQ()) {
					int rows = e.length();
					int cols = e.part(1).length();
					m.setRowCount(rows);
					m.setColumnCount(cols);
					for (int i = 0; i < rows; ++i) {
						Expr e2 = e.part(i+1);
						for (int j = 0; j < cols; ++i) {
						  m.setValueAt( convertExprAsContent( e2.part(j+1)), i, j);
						  }
						e2.dispose();
						}
					}
			
				return m;
				}
					
			// TODO sometime we can check if the requested class is
			// immutable or not and return an instance that matches 
			// this access
			else if (TreeModel.class.isAssignableFrom(to)) {
			  DefaultMutableTreeNode root = createTreeNode((Expr)obj);
				DefaultTreeModel m = new DefaultTreeModel(root);
				return m;
				}
					
			// TODO sometime we can check if the requested class is
			// immutable or not and return an instance that matches 
			// this access
			else if (TreeNode.class.isAssignableFrom(to)) {
				DefaultMutableTreeNode root = createTreeNode((Expr)obj);
				return root;
				}
        
			else {
				return null;
				}
			}
		 catch(Exception e) {return null;}
			}
		else
			return null;
	}

  public Object convertExprAsContent(Expr e) {
    try {
    	if (e.bigDecimalQ()) {
    		return MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(Expr.class, e, BigDecimal.class);
    		}
  		else if (e.bigIntegerQ()) {
  			return MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(Expr.class, e, BigInteger.class);
  			}
  		else if (e.stringQ()) {
  			return MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(Expr.class, e, String.class);
  			}
  		else if (e.equals(Expr.SYM_TRUE) || e.equals(Expr.SYM_FALSE)) {
  			return MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(Expr.class, e, Boolean.class);
  			}
  		else if (e.realQ()) {
  			return MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(Expr.class, e, Double.class);
       }
  		else if (e.integerQ()) {
  			return MathematicaTypeConvertorRegistry.typeConvertorRegistry.convertAsObject(Expr.class, e, Integer.class);
  			}
      }
    catch (MathematicaBSFException me) {
      return null;
      }
  	return e;
  	}
  	
  // Not implemented or used anywhere
  public String getCodeGenString () {
    return "";
    }
 
  protected void putListContent(MathLink link, Object element) throws MathLinkException {
		//	TODO currently having to use put() forces non JLink converted
		// Mathematica types to Strings instead of being able to do putReference()
		// on a KernelLink to preserve some elements as JavaObjects if more complex
		link.put( element);
  	}
  	
  protected void putTableCellContent(MathLink link, Object cell) throws MathLinkException {
		//	TODO currently having to use put() forces non JLink converted
		// Mathematica types to Strings instead of being able to do putReference()
		// on a KernelLink to preserve some elements as JavaObjects if more complex
		link.put( cell);
  	}
  	
  protected DefaultMutableTreeNode createTreeNode(Expr e) {
  	if (e == null) return new DefaultMutableTreeNode(null);
  	if (e.listQ() && e.length() == 2 && e.part(2).listQ()) {
			DefaultMutableTreeNode thisNode = new DefaultMutableTreeNode(convertExprAsContent(e.part(1)));
			int count = e.part(2).length();
  		for (int i = 0; i < count; ++i) {
				DefaultMutableTreeNode childNode = createTreeNode(e.part(2).part(i+1));
				thisNode.add(childNode);
  			}
  		return thisNode;
  		}
  	else {
  		return new DefaultMutableTreeNode( convertExprAsContent(e),  false);
  		}
    }
  
	protected void putTreeNodeContent(MathLink link, Object node) throws MathLinkException {
		//	TODO currently having to use put() forces non JLink converted
		// Mathematica types to Strings instead of being able to do putReference()
		// on a KernelLink to preserve some elements as JavaObjects if more complex
		if (node instanceof DefaultMutableTreeNode) {
			link.put( ((DefaultMutableTreeNode)node).getUserObject());
			}
		else link.put( node);
		}
		
	protected void putTreeNode(MathLink link, TreeNode node) throws MathLinkException {
		if (node == null) {
			link.putSymbol("Null");
			}
		else if (node instanceof TreeNode && !((TreeNode)node).getAllowsChildren()) {
			putTreeNodeContent(link, node);
			}
		else {
			link.putFunction("List", 2);
			putTreeNodeContent(link, node);
			int count = node.getChildCount();
			link.putFunction("List", count);
			for (int i = 0; i < count; ++i) {
				putTreeNode(link, node.getChildAt(i));
				}	
			}
		}
		
  protected void putTreeNode(MathLink link, TreeModel model, Object node) throws MathLinkException {
  	if (node == null) {
  		link.putSymbol("Null");
  		}
  	else if (node instanceof TreeNode && !((TreeNode)node).getAllowsChildren()) {
  		putTreeNodeContent(link, node);
  		}
  	else {
			link.putFunction("List", 2);
			putTreeNodeContent(link, node);
			int count = model.getChildCount(node);
			link.putFunction("List", count);
			for (int i = 0; i < count; ++i) {
				putTreeNode(link, model, model.getChild(node, i));
				}	
  		}
  	}
  
}

