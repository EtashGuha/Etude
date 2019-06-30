/*
 * @(#)DefaultNodeController.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
 /*
 * based on a diva class
 * Copyright (c) 1998-2001 The Regents of the University of California.
 * All rights reserved. See the file diva-COPYRIGHT.txt for details.
 */
package com.wolfram.guikit.graph;

import java.awt.geom.Point2D;
import java.util.ArrayList;
import java.util.Iterator;

import diva.canvas.Figure;
import diva.canvas.connector.Connector;
import diva.canvas.connector.ConnectorEvent;
import diva.canvas.interactor.SelectionModel;

import diva.graph.BasicNodeController;
import diva.graph.GraphController;
import diva.graph.MutableGraphModel;

/**
 * DefaultNodeController
 *
 * @version	$Revision: 1.1 $
 */
public class DefaultNodeController extends BasicNodeController {

  public DefaultNodeController (GraphController controller) {
    super(controller);    
    }
     
  /** 
   * Remove the figure for the given node.
   * But in this subclass we disconnect edges before clear
   * so they stick around
   */
  public void clearNodeOnly(Object node) {
    MutableGraphModel model = (MutableGraphModel)getController().getGraphModel();
    ArrayList disconnectList = new ArrayList();
    SelectionModel sm = getController().getSelectionModel(); 
    for(Iterator i = model.outEdges(node); i.hasNext(); ) {
      disconnectList.add(i.next());
      }
    for(Iterator i = disconnectList.iterator(); i.hasNext(); ) {
      Object edge = i.next();
      Figure edgeFigure = getController().getFigure(edge);
      boolean selected = sm.containsSelection(edgeFigure);
      if (selected) {
         sm.removeSelection(edgeFigure);
         }
      Point2D nodeOrigin = ((Connector)edgeFigure).getTailSite().getPoint();
      Object otherNode = model.getHead(edge);
      if (otherNode == null || otherNode.equals(node)) {
        Point2D nodeEnd = ((Connector)edgeFigure).getHeadSite().getPoint();
        getController().clearEdge(edge);
        model.disconnectEdge(getController(), edge);
        ((DefaultEdgeController)getController().getEdgeController(edge)).addEdge(edge, 
           nodeOrigin.getX(), nodeOrigin.getY(), nodeEnd.getX(), nodeEnd.getY());
        }
      else {
        getController().clearEdge(edge);
        model.disconnectEdge(getController(), edge);
        getController().addEdge(edge, otherNode, ConnectorEvent.HEAD_END, 
          nodeOrigin.getX(), nodeOrigin.getY());
        }
       if (selected) {
         sm.addSelection(getController().getFigure(edge));
         }
      }
    disconnectList.clear();
    for(Iterator i = model.inEdges(node); i.hasNext(); ) {
      disconnectList.add(i.next());
      } 
    for(Iterator i = disconnectList.iterator(); i.hasNext(); ) {
      Object edge = i.next();
      Figure edgeFigure = getController().getFigure(edge);
      boolean selected = sm.containsSelection(edgeFigure);
      if (selected) {
        sm.removeSelection(edgeFigure);
        }
      Point2D nodeOrigin = ((Connector)edgeFigure).getHeadSite().getPoint();
      Object otherNode = model.getTail(edge);
      if (otherNode == null) {
        Point2D nodeEnd = ((Connector)edgeFigure).getTailSite().getPoint();
        getController().clearEdge(edge);
        model.disconnectEdge(getController(), edge);
        ((DefaultEdgeController)getController().getEdgeController(edge)).addEdge(edge, 
           nodeEnd.getX(), nodeEnd.getY(), nodeOrigin.getX(), nodeOrigin.getY());
        }
      else {
        getController().clearEdge(edge);
        model.disconnectEdge(getController(), edge);
        getController().addEdge(edge, otherNode, ConnectorEvent.TAIL_END, 
          nodeOrigin.getX(), nodeOrigin.getY());
        }
      if (selected) {
        sm.addSelection(getController().getFigure(edge));
        }
      }
    
    super.clearNode(node);
    }
    
    /** 
     * Remove the node.
     */
    public void removeNodeOnly(Object node) {
      // FIXME why isn't this symmetric with addNode?
      MutableGraphModel model = (MutableGraphModel) getController().getGraphModel();
      // clearing the nodes is responsible for clearing any edges that are
      // connected
      if(model.isComposite(node)) {
        for(Iterator i = model.nodes(node); i.hasNext(); ) {
          Object insideNode = i.next();
          if (getController() instanceof DefaultGraphController)
            ((DefaultGraphController)getController()).clearNodeOnly(insideNode);
          else getController().clearNode(insideNode);
         }
      }
      clearNodeOnly(node);
      // we assume that the model will remove any edges that are connected.
      model.removeNode(getController(), node);
      getController().getGraphPane().repaint();
      }
     
}



