/*
 * @(#)DefaultGraphController.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
 /*
 * based on a diva class
 * Copyright (c) 1998-2001 The Regents of the University of California.
 * All rights reserved. See the file diva-COPYRIGHT.txt for details.
 */
package com.wolfram.guikit.graph;

import diva.graph.*;
import diva.graph.basic.BasicGraphModel;
import diva.util.PropertyContainer;

import diva.canvas.event.LayerEvent;
import diva.canvas.event.MouseFilter;
import diva.canvas.interactor.AbstractInteractor;
import diva.canvas.interactor.Interactor;
import diva.canvas.interactor.SelectionInteractor;
import diva.canvas.interactor.SelectionDragger;

import java.awt.event.InputEvent;

import com.wolfram.guikit.diva.CompoundMouseFilter;

/**
 * DefaultGraphController is a basic implementation of GraphController, which works with
 * simple graphs that have edges connecting simple nodes. It
 * sets up some simple interaction on its view's pane.
 *
 * @version	$Revision: 1.1 $
 */
public class DefaultGraphController extends SimpleGraphController {
    /**
     * The global count for the default node/edge creation.
     */
    private int _globalCount = 0;

    /** The selection interactor for drag-selecting nodes
     */
    private SelectionDragger _selectionDragger;

   /** The interactor for creating new nodes
     */
    private NodeCreator _nodeCreator;

    /** The interactor that interactively creates edges
     */
    private EdgeCreator _edgeCreator;

    /** The filter for creating nodes or edges
     */
    private MouseFilter _createFilter = new CompoundMouseFilter(
      new MouseFilter[]{
        new MouseFilter(InputEvent.BUTTON2_MASK|InputEvent.BUTTON3_MASK),
        new MouseFilter(InputEvent.BUTTON1_MASK|InputEvent.BUTTON2_MASK|InputEvent.BUTTON3_MASK,
           InputEvent.CTRL_MASK)
        });

    /**
     * Create a new basic controller with default node and edge controllers.
     */
    public DefaultGraphController () {
      NodeController nc = new DefaultNodeController(this);
      nc.setNodeRenderer(new DefaultNodeRenderer(this));
      setNodeController(nc);
      
      DefaultEdgeController ec = new DefaultEdgeController(this);
      ec.setConnectorTarget(new DefaultConnectorTarget());
      ec.setEdgeRenderer(new DefaultEdgeRenderer(this));
      setEdgeController(ec);
      
      // addGraphViewListener(new IncrementalLayoutListener( new IncrLayoutAdapter(new LevelLayout(new BasicLayoutTarget(this))), null));
      }

    public Interactor getNodeCreator() {return _nodeCreator;}
    public Interactor getEdgeCreator() {return _edgeCreator;}
    
    /** 
     * Remove the figure for the given node.
     */
    public void clearNodeOnly(Object node) {
      NodeController nc = getNodeController(node);
      if (nc instanceof DefaultNodeController) ((DefaultNodeController)nc).clearNodeOnly(node);
      else nc.clearNode(node);
      }
    
    /**
     * Remove the given node.  Find the node controller associated with that
     * node and delegate to that node controller.
     */
    public void removeNodeOnly(Object node) {
      if (getNodeController(node) instanceof DefaultNodeController)
         ((DefaultNodeController)getNodeController(node)).removeNodeOnly(node);
      else getNodeController(node).removeNode(node);
      }
    
    /**
     * Initialize all interaction on the graph pane. This method
     * is called by the setGraphPane() method of the superclass.
     * This initialization cannot be done in the constructor because
     * the controller does not yet have a reference to its pane
     * at that time.
     */
  protected void initializeInteraction () {
		final GraphPane pane = getGraphPane();

        // Create and set up the selection dragger
        _selectionDragger = new SelectionDragger(pane);
        _selectionDragger.addSelectionInteractor(
                (SelectionInteractor)getEdgeController().getEdgeInteractor());
				_selectionDragger.addSelectionInteractor(
                (SelectionInteractor)getNodeController().getNodeInteractor());

        // Create a listener that creates new nodes
        _nodeCreator = new NodeCreator();
        _nodeCreator.setMouseFilter(_createFilter);
        pane.getBackgroundEventLayer().addInteractor(_nodeCreator);

		// Create the interactor that drags new edges.
		_edgeCreator = new EdgeCreator(this) {
	    public Object createEdge() {	
        getSelectionModel().clearSelection();
	    	if (pane instanceof ExprAccessibleGraphPane) {
	    		return ((ExprAccessibleGraphPane)pane).createEdge((BasicGraphModel)getGraphModel(),  new Integer(_globalCount++),
	    		   null, (PropertyContainer) getGraphModel().getRoot());
	    	  }
	    	else {
				  Object semanticObject = new Integer(_globalCount++);
				  BasicGraphModel bgm = (BasicGraphModel)getGraphModel();
				  return bgm.createEdge(semanticObject);
	    	}
	    	}
			};
			
		_edgeCreator.setMouseFilter(_createFilter);
		((NodeInteractor)getNodeController().getNodeInteractor()).addInteractor(_edgeCreator);
		
    }

    ///////////////////////////////////////////////////////////////
    //// NodeCreator

    /** An inner class that places a node at the clicked-on point
     * on the screen, This
     * needs to be made more customizable.
     */
    protected class NodeCreator extends AbstractInteractor {
      public void mousePressed(LayerEvent e) {
        getSelectionModel().clearSelection();
        Object semanticObject = new Integer(_globalCount++);
        BasicGraphModel bgm = (BasicGraphModel)getGraphModel();
        Object node = bgm.createNode(semanticObject);
        addNode(node,  e.getLayerX(), e.getLayerY());
        getSelectionModel().addSelection(getFigure(node));
        }
      }
}



