/*
 * @(#)DefaultGraphModel.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.graph;

import diva.graph.basic.BasicGraphModel;
import diva.graph.modular.BasicModularGraphModel;
import diva.graph.modular.Graph;

/**
 * DefaultGraphModel
 *
 * @version	$Revision: 1.1 $
 */
public class DefaultGraphModel extends BasicGraphModel {
    
	public DefaultGraphModel() {
	  super();
	  }
	  
	public DefaultGraphModel(Graph root) {
		new BasicModularGraphModel(root);
		}
	
}



