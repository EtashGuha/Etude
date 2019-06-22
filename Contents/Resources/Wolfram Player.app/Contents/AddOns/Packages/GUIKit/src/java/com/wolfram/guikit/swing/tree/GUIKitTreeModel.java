/*
 * @(#)GUIKitTreeModel.java 1.14 03/01/23
 */
package com.wolfram.guikit.swing.tree;

import javax.swing.tree.DefaultTreeModel;
import javax.swing.tree.TreeNode;

/**
 * GUIKitTreeModel
 */
public class GUIKitTreeModel extends DefaultTreeModel {
  
  private static final long serialVersionUID = -1287287975456782248L;
  
  public GUIKitTreeModel() {
    this(null);
    }
    
  public GUIKitTreeModel(TreeNode root) {
    super(root);
    }

  public GUIKitTreeModel(TreeNode root, boolean asksAllowsChildren) {
    super(root, asksAllowsChildren);
    }
  
  public void setRoot(Object root) {
    if (root == null ||  root instanceof TreeNode) super.setRoot((TreeNode)root);
    }
    
}

