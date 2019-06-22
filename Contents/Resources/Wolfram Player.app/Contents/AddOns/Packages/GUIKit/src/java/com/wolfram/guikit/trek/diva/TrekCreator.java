/*
 * @(#)TrekController.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.trek.diva;

import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;

import diva.canvas.event.LayerEvent;
import diva.canvas.event.LayerEventMulticaster;
import diva.canvas.event.LayerListener;
import diva.canvas.interactor.AbstractInteractor;
import diva.canvas.interactor.SelectionInteractor;

/** 
 * TrekCreator places a node at the clicked-on point
 * on the screen, This needs to be made more customizable.
 */
public class TrekCreator extends AbstractInteractor {
  /** Layer listeners */
  private transient LayerListener _layerListener;
        
  /* The most recent coordinates */
  private double _prevX = 0.0;
  private double _prevY = 0.0;

  /* Enable only if the figure in the event is in the selection */
  private boolean _selectiveEnabled;
    
  private TrekPane trekPane;
    
  public TrekCreator(TrekPane trekPane) {
    super();
    this.trekPane = trekPane;
    }
    
      /** Add the given layer listener to this interactor.  Any event that is
     * received by this interactor will be passed on to the listener after
     * it is handled by this interactor.
     */
    public void addLayerListener(LayerListener l) {
        _layerListener = LayerEventMulticaster.add(_layerListener,l);
    }
    
  
    /** Remove the given layer listener from this interactor.
     */
    public void removeLayerListener(LayerListener l) {
        _layerListener = LayerEventMulticaster.remove(_layerListener, l);
    }
    
        /** Fire a layer event.
     */
    public void fireLayerEvent (LayerEvent event) {
        if (_layerListener != null) {
            int id = event.getID();
            switch(id) {
            case MouseEvent.MOUSE_PRESSED:
                _layerListener.mousePressed(event);
                break;
            case MouseEvent.MOUSE_DRAGGED:
                _layerListener.mouseDragged(event);
                break;
            case MouseEvent.MOUSE_RELEASED:
                _layerListener.mouseReleased(event);
                break;
            }
        }
    }
    
    
     /** Constrain the point and move the target if the mouse
     * move. The target movement is done by the translate()
     * method, which can be overridden to change the behaviour.
     * Nothing happens if the interactor is not enabled, or if it
     * is "selective enabled" but not in the selection.
     */
    public void mouseDragged (LayerEvent e) {
        if (!isEnabled()
                || (_selectiveEnabled && !SelectionInteractor.isSelected(e))) {
            return;
        }
        if(getMouseFilter() == null || getMouseFilter().accept(e)) {
        	
					if (!trekPane.getCanvas().getCursor().equals(TrekController.CREATE_CURSOR))
									trekPane.getCanvas().setCursor(TrekController.CREATE_CURSOR);
									
            // Constrain the point
            Point2D p = e.getLayerPoint();
            // Translate and consume if the point changed
            double x = p.getX();
            double y = p.getY();
            double deltaX = x - _prevX;
            double deltaY = y - _prevY;
            if (deltaX != 0 || deltaY != 0) {
                fireLayerEvent(e);
            }
            _prevX = x;
            _prevY = y;
        
            // Consume the event
            if (isConsuming()) {
                e.consume();
            }
        }
    }

    /** Handle a mouse press on a figure or layer. Set the target
     * to be the figure contained in the event. Call the setup()
     * method in case there is additional setup to do, then
     * constrain the point and remember it. 
     * Nothing happens if the interactor is not enabled, or if it
     * is "selective enabled" but not in the selection.
     */
    public void mousePressed (LayerEvent e) {
        if (!isEnabled()
                || (_selectiveEnabled && !SelectionInteractor.isSelected(e))) {
            return;
        }
        if(getMouseFilter() == null || getMouseFilter().accept(e)) {
        	
        		trekPane.getCanvas().setCursor(TrekController.CREATE_CURSOR);
        		
            // Constrain and remember the point
            Point2D p = e.getLayerPoint();
            // FIXME: no, don't constrain in mouse-pressed!?
            //constrainPoint(p);
            _prevX = p.getX();
            _prevY = p.getY();
            
            // Inform listeners
            fireLayerEvent(e);
            
            // Consume the event
            if (isConsuming()) {
                e.consume();
            }
        }
    }

    /** Set the flag that says that the interactor responds only
     * if the figure being moused on is selected. By default, this
     * flag is false; if set true, then the mouse methods check that
     * the figure is contained in the selection model of that
     * figure's selection interactor (if it has one).
     */
    public boolean setSelectiveEnabled (boolean s) {
        return _selectiveEnabled = s;
    }

    /** Get the flag that says that the interactor responds only
     * if the figure being moused on is selected. By default, this
     * flag is false.
     */
    public boolean getSelectiveEnabled () {
        return _selectiveEnabled;
    }
    /** Handle a mouse released event.
     * Nothing happens if the interactor is not enabled, if if it
     * is "selective enabled" but not in the selection.
     */
    public void mouseReleased (LayerEvent e) {
        if (!isEnabled()
                || (_selectiveEnabled && !SelectionInteractor.isSelected(e))) {
            return;
        }
        if(getMouseFilter() == null || getMouseFilter().accept(e)) {
        	  trekPane.updateCursor();
        	  
            fireLayerEvent(e);
   
            // Consume the event
            if (isConsuming()) {
                e.consume();
            }
        }
    }
    
}



