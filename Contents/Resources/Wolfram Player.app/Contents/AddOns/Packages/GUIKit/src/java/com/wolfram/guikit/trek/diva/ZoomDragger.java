/*
 * @(#)ZoomDragger.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
 /*
 * based on a diva class
 * Copyright (c) 1998-2001 The Regents of the University of California.
 * All rights reserved. See the file diva-COPYRIGHT.txt for details.
 */
package com.wolfram.guikit.trek.diva;

import diva.canvas.OverlayLayer;
import diva.canvas.event.EventLayer;
import diva.canvas.event.LayerEvent;
import diva.canvas.event.MouseFilter;
import diva.canvas.interactor.DragInteractor;

import java.awt.Dimension;
import java.awt.geom.Rectangle2D;

/**
 * ZoomDragger is a class that implements rubber-banding on a canvas.
 * 
 * @version $Revision: 1.5 $
 */
public class ZoomDragger extends DragInteractor {

    /* The overlay layer
     */
    private OverlayLayer _overlayLayer;

    /* The event layer
     */
    private EventLayer _eventLayer;

    /* The rubber-band
     */
    private Rectangle2D _rubberBand = null;

    /* The origin points
     */
    private double _originX;
    private double _originY;
		private double prevX;
		private double prevY;
    /** The mouse filter for selecting items
     */
    private MouseFilter _zoomInFilter = MouseFilter.selectionFilter;

    /** The mouse filter for toggling items
     */
    private MouseFilter _zoomOutFilter = MouseFilter.alternateSelectionFilter;

    /** The selection mode flags
     */
    private boolean _isZoomingIn;
    private boolean _isZoomingOut;
		private boolean isMouseDown = false;
		
		private TrekPane trekPane;
		
    ///////////////////////////////////////////////////////////////////
    ////                         constructors                      ////

    /**
     * Create a new SelectionDragger attached to the given graphics
     * pane.
     */
    public ZoomDragger(TrekPane gpane) {
      super();
      this.trekPane = gpane;
      setOverlayLayer(gpane.getOverlayLayer());
      setEventLayer(gpane.getBackgroundEventLayer());
    	}

    /**
     * Get the layer that drag rectangles are drawn on
     */
    public OverlayLayer getOverlayLayer () {
			return _overlayLayer;
    	}

    /**
     * Get the layer that drag events are listened on
     */
    public EventLayer getEventLayer () {
			return _eventLayer;
    	}

    /**
     * Get the mouse filter that controls when this selection
     * filter is activated.
     */
    public MouseFilter getZoomInFilter () {
        return _zoomInFilter;
    }

    /**
     * Get the mouse filter that controls the toggling of
     * selections
     */
    public MouseFilter getZoomOutFilter () {
			return _zoomOutFilter;
    	}

    /** Reshape the rubber-band, swapping coordinates if necessary.
     * Any figures that are newly included or excluded from
     * the drag region are added to or removed from the appropriate
     * selection.
     */
    public void mouseDragged (LayerEvent event) {
 			if (!isEnabled()) return;
			if (!_isZoomingOut && !_isZoomingIn) return;
        
			if (_rubberBand == null) {
	    	// This should never happen, but it does.
	    	return;
				}
      
			if (_zoomInFilter.accept(event) || _zoomOutFilter.accept(event)) {
				_isZoomingIn = _zoomInFilter.accept(event);
		  	_isZoomingOut = _zoomOutFilter.accept(event);
				}
			
			trekPane.getCanvas().setCursor( _isZoomingIn ? TrekController.ZOOMIN_CURSOR : TrekController.ZOOMOUT_CURSOR);
			
			// Figure out the coordinates of the rubber band
			_overlayLayer.repaint(_rubberBand);

		  double useXPos = event.getLayerX();
		  double useYPos = event.getLayerY();
		  
			// space held down translates
			// rubberband _originX, _originY by event-prev
			if (((TrekCanvas)trekPane.getCanvas()).isSpaceDown()) {
				_originX += (useXPos - prevX);
				_originY += (useYPos - prevY);
				}
				
			prevX = event.getLayerX();
			prevY = event.getLayerY();
			
		  // Without control down rubberband conforms to visible aspect ratio
		  if (!((TrekCanvas)trekPane.getCanvas()).isControlDown()) {
				Dimension visSize = trekPane.getVisibleSize();
				double currAspect = visSize.getHeight()/visSize.getWidth();
				if (Math.abs(useYPos - _originY) > Math.abs(useXPos - _originX)) {
					useXPos = _originX + (useXPos < _originX ? -1.0 : 1.0)*Math.abs((useYPos - _originY)/currAspect);
					}
				else {
					useYPos = _originY + (useYPos < _originY ? -1.0 : 1.0)*Math.abs((useXPos - _originX)*currAspect);
					}
		  	}
		  
			if (event.isAltDown()) {
				_rubberBand.setFrameFromCenter(_originX, _originY, useXPos, useYPos);
				}
      else {
				_rubberBand.setFrameFromDiagonal(_originX, _originY, useXPos, useYPos);
      	}
      	
			_overlayLayer.repaint(_rubberBand);

			// Could possibly do some custom drawing showing an in or out zoom
			if (_isZoomingIn) {
        } 
      else {
        }
        
		// Consume the event
		if (isConsuming()) {
			event.consume();
			}
	}

    /** Clear the selection, and create the rubber-band
     */
	public void mousePressed (LayerEvent event) {
		isMouseDown = true;
		if (!isEnabled()) return;
  
		// Check mouse event, set flags, etc
		_isZoomingIn = _zoomInFilter.accept(event);
		_isZoomingOut = _zoomOutFilter.accept(event);

		if (!_isZoomingOut && !_isZoomingIn) return;

		// Do it
  	_originX = event.getLayerX();
    _originY = event.getLayerY();
		prevX = event.getLayerX();
		prevY = event.getLayerY();
		
    _rubberBand = new Rectangle2D.Double(_originX,_originY, 0.0, 0.0);

		_overlayLayer.add(_rubberBand);
		_overlayLayer.repaint(_rubberBand);

		trekPane.getCanvas().setCursor( _isZoomingIn ? TrekController.ZOOMIN_CURSOR : TrekController.ZOOMOUT_CURSOR);
		
		// Consume the event
		if (isConsuming()) {
	    event.consume();
			}
		}

    /** Delete the rubber-band
     */
	public void mouseReleased (LayerEvent event) {
		isMouseDown = false;
		if (!isEnabled()) return;

		if (_rubberBand == null) {
	    // This should never happen, but it does.
	    return;
			}

		trekPane.updateCursor();
		
		terminateDrag(event);

		// Consume the event
		if (isConsuming()) {
	    event.consume();
			}
	}


    /**
     * Set the layer that drag rectangles are drawn on
     */
	public void setOverlayLayer (OverlayLayer l) {
		_overlayLayer = l;
    }

    /**
     * Set the layer that drag events are listened on
     */
	public void setEventLayer (EventLayer l) {
		if (_eventLayer != null) {
			_eventLayer.removeLayerListener(this);
			}
  	_eventLayer = l;
		_eventLayer.addLayerListener(this);
    }

    /**
     * Set the mouse filter that controls when this selection
     * filter is activated.
     */
    public void setZoomInFilter(MouseFilter f) {
        _zoomInFilter = f;
    }

    /**
     * Set the mouse filter that controls the toggling of
     * selections.
     */
	public void setZoomOutFilter(MouseFilter f) {
		_zoomOutFilter = f;
    }


	public void zoom(double centerX, double centerY, double zoomAmount) {
		double newCenterX = trekPane.getTrekCoordinateX(centerX);
		double newCenterY = trekPane.getTrekCoordinateY(centerY);
		trekPane.addScaleFactor(zoomAmount, zoomAmount, newCenterX, newCenterY);
		}

	public void scale(double centerX, double centerY, double scaleXAmount, double scaleYAmount ) {
		double newCenterX = trekPane.getTrekCoordinateX(centerX);
		double newCenterY = trekPane.getTrekCoordinateY(centerY);
		trekPane.addScaleFactor(scaleXAmount, scaleYAmount, newCenterX, newCenterY);
		}
		
  public boolean isMouseDown() {
  	return isMouseDown;
  	}
  
    /** Terminate drag-selection operation. This must only be called
     * from events that are triggered during a drag operation.
     */
	public void terminateDrag(LayerEvent event) {
		if (!_isZoomingOut && !_isZoomingIn) return;

		if (_zoomInFilter.accept(event) || _zoomOutFilter.accept(event)) {
			_isZoomingIn = _zoomInFilter.accept(event);
			_isZoomingOut = _zoomOutFilter.accept(event);
			}

			
		// Use the positions of the rubberband to perform the zoom
		if( _rubberBand.getHeight() < 2.0 || _rubberBand.getWidth() < 2.0) {
			//We treat this as a single click zoom at center
			if (_isZoomingIn) {
				// TODO make the click factor zooming settable instead of 125%/80% which is a good default
				zoom(_rubberBand.getCenterX(), _rubberBand.getCenterY(), 1.25);
				} 
			else if (_isZoomingOut) {
				zoom(_rubberBand.getCenterX(), _rubberBand.getCenterY(), 0.8);
				}
			}
		else {
			
			// We treat this as the scale to fit the bounds of these converted coordinates
			if (((TrekCanvas)trekPane.getCanvas()).isControlDown()) {
				// We treat this as the zoom to fit the bounds of these converted coordinates
				Dimension visSize = trekPane.getVisibleSize();
				double xZoomAmount = visSize.getWidth()/_rubberBand.getWidth();
				double yZoomAmount = visSize.getHeight()/_rubberBand.getHeight();
			  if (_isZoomingOut) {
			  	xZoomAmount = 1.0/xZoomAmount;
					yZoomAmount = 1.0/yZoomAmount;
					}
				scale(_rubberBand.getCenterX(), _rubberBand.getCenterY(), xZoomAmount, yZoomAmount);
				}
			else {
				// We treat this as the zoom to fit the bounds of these converted coordinates
				Dimension visSize = trekPane.getVisibleSize();
				double currAspect = visSize.getHeight()/visSize.getWidth();
				double zoomAmount = 1.0;
			  if (currAspect >= 1.0) {
				  zoomAmount = visSize.getHeight()/_rubberBand.getHeight();
					if (_isZoomingOut) zoomAmount = 1.0/zoomAmount;
					}
				else {
					zoomAmount = visSize.getWidth()/_rubberBand.getWidth();
					if (_isZoomingOut) zoomAmount = 1.0/zoomAmount;
					}
				zoom(_rubberBand.getCenterX(), _rubberBand.getCenterY(), zoomAmount);
				}
			}
		
		
		_overlayLayer.repaint(_rubberBand);
		_overlayLayer.remove(_rubberBand);
		_rubberBand = null;
		}
    	
}



