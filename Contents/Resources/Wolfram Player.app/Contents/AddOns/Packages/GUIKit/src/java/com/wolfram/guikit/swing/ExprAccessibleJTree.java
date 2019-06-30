/*
 * @(#)ExprAccessibleListModel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.swing;

import java.util.Vector;
import java.util.Hashtable;

import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreeNode;
import javax.swing.JTree;
import javax.swing.tree.TreeModel;

import com.wolfram.bsf.util.type.ExprTypeConvertor;
import com.wolfram.guikit.swing.tree.GUIKitTreeModel;
import com.wolfram.guikit.type.ExprAccessible;

import com.wolfram.jlink.Expr;

/**
 * ExprAccessibleListModel extends DefaultListModel
 */
public class ExprAccessibleJTree extends JTree implements ExprAccessible {

    private static final long serialVersionUID = -1277987975456778938L;
    
	private static final ExprTypeConvertor convertor = new ExprTypeConvertor();
	
	public ExprAccessibleJTree() {
		this(getDefaultTreeModel());
		}
    
	public ExprAccessibleJTree(Object[] value) {
		super(value);
		}
	public ExprAccessibleJTree(Hashtable value) {
		super(value);
		}
	public ExprAccessibleJTree(Vector value) {
		super(value);
		}
	public ExprAccessibleJTree(TreeModel newModel) {
		super(newModel);
		}
	public ExprAccessibleJTree(TreeNode root) {
		super(root);
		}
	public ExprAccessibleJTree(TreeNode root, boolean asksAllowsChildren) {
		super(root, asksAllowsChildren);
		}
		
  /**
   * Creates and returns a sample <code>TreeModel</code>.
   * Used primarily for beanbuilders to show something interesting.
   *
   * @return the default <code>TreeModel</code>
   */
  protected static TreeModel getDefaultTreeModel() {
    DefaultMutableTreeNode root = new DefaultMutableTreeNode("Tree");
    DefaultMutableTreeNode parent;
  
    parent = new DefaultMutableTreeNode("colors");
    root.add(parent);
    parent.add(new DefaultMutableTreeNode("blue"));
    parent.add(new DefaultMutableTreeNode("violet"));
    parent.add(new DefaultMutableTreeNode("red"));
    parent.add(new DefaultMutableTreeNode("yellow"));
  
    parent = new DefaultMutableTreeNode("sports");
    root.add(parent);
    parent.add(new DefaultMutableTreeNode("basketball"));
    parent.add(new DefaultMutableTreeNode("soccer"));
    parent.add(new DefaultMutableTreeNode("football"));
    parent.add(new DefaultMutableTreeNode("hockey"));
  
    parent = new DefaultMutableTreeNode("food");
    root.add(parent);
    parent.add(new DefaultMutableTreeNode("hot dogs"));
    parent.add(new DefaultMutableTreeNode("pizza"));
    parent.add(new DefaultMutableTreeNode("ravioli"));
    parent.add(new DefaultMutableTreeNode("bananas"));
    return new GUIKitTreeModel(root);
    }
    
	public Expr getExpr() {
	  Object result = convertor.convert(getModel().getClass(), Expr.class, getModel());
		return (result instanceof Expr) ? (Expr)result : null;
		}
		
	public void setExpr(Expr e) {
		Object result = convertor.convert(Expr.class, getModel().getClass(), e);
		if (result != null && result instanceof TreeModel)
			setModel((TreeModel)result);
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
		// TODO implement
		Object result = convertor.convert(Expr.class, TreeNode.class, e);
		if (result != null && result instanceof TreeNode) {
			// if index is valid we can insert TreeNode result within proper place in tree model
			// which could include negative indexing to append or fill?
		  }
		}
		
	public void setPart(int[] ia, Expr e) {
		// TODO implement
		Object result = convertor.convert(Expr.class, TreeNode.class, e);
		if (result != null && result instanceof TreeNode) {
			// if index is valid we can insert TreeNode result within proper place in tree model
			//		which could include negative indexing to append or fill?
			}
		}
	
  }
