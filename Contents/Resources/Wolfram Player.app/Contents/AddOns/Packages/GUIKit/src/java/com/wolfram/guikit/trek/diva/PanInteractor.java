/*
 * $Id: PanInteractor.java,v 1.2 2004/04/02 16:55:22 jeffa Exp $
 *
 */
package com.wolfram.guikit.trek.diva;

import diva.canvas.event.EventLayer;
import diva.canvas.event.LayerEvent;
import diva.canvas.event.MouseFilter;
import diva.canvas.interactor.AbstractInteractor;

/** 
 * PanInteractor
 *
 * @version $Revision: 1.2 $
 */
public class PanInteractor extends AbstractInteractor {

    /** The mouse filter for selecting items
     */
    private MouseFilter _panFilter = MouseFilter.selectionFilter;

    private TrekPane trekPane;
    
	  private boolean dragging;
		private double prevX, prevY;
		 
    /* The event layer
     */
    private EventLayer _eventLayer;

    ///////////////////////////////////////////////////////////////////
    ////                         constructors                      ////

    /**
     * Create a new SelectionInteractor with a default selection model and
     * a default selection renderer.
     */
    public PanInteractor(TrekPane gpane) {
      super();
      trekPane = gpane;
      setEventLayer(gpane.getBackgroundEventLayer());
      }

    ///////////////////////////////////////////////////////////////////
    //// public methods

    /**
     * Accept an event if it will be accepted by the selection
     * filters.
     */
    public boolean accept (LayerEvent e) {
      return _panFilter.accept(e) || super.accept(e);
      }

    public MouseFilter getPanFilter () {
      return _panFilter;
      }

    /** Handle a mouse press event. Add or remove the clicked-on
     * item to or from the selection. If it's still in the selection,
     * pass the event to the superclass to handle.
     */
    public void mousePressed(LayerEvent event) {
        if (!isEnabled()) {
     		  return;
        	}
 
			  dragging = false;
        
        trekPane.getCanvas().setCursor(TrekController.PAN_CURSOR);
        
        if (!_panFilter.accept(event)) return;
         
        trekPane.centerAt(
            trekPane.getTrekCoordinateX(event.getLayerX()), 
            trekPane.getTrekCoordinateY(event.getLayerY()));
        
				prevX = event.getLayerX();
				prevY = event.getLayerY();
				
			  dragging = true;
        // Allow superclass to process event
        super.mousePressed(event);

        // Always consume the event if the pan occurred, regardless of the consuming flag
        if (dragging) {
            event.consume();
        }
    }

	public void mouseDragged(LayerEvent event) {
			if (!isEnabled()) {
					return;
			}
 
			if (!dragging)
				return;
        
			if (!_panFilter.accept(event)) return;
		
      trekPane.getCanvas().setCursor(TrekController.PAN_CURSOR);
      
		  if (event.getLayerX() == prevX && event.getLayerY() == prevY)
				return;
					
		  // call a translate by..
		  trekPane.translateBy( 
		  	trekPane.getTrekCoordinateX(event.getLayerX()) - trekPane.getTrekCoordinateX(prevX),
				trekPane.getTrekCoordinateY(event.getLayerY()) - trekPane.getTrekCoordinateY(prevY)
		  	);
		  
		  
			prevX = event.getLayerX();
			prevY = event.getLayerY();
			
			// Allow superclass to process event
			super.mouseDragged(event);

			// Always consume the event if the pan occurred, regardless of the consuming flag
			if (dragging) {
				event.consume();
				}
	}
	
	/**
	 *  Responds to a mouse-release.  Not meant to be called directly.
	 */   
	public void mouseReleased(LayerEvent event) {
		 if (!dragging)
				return;
				
			event.consume();
			mouseDragged(event);
			dragging = false;

      trekPane.updateCursor();

			}
	
    
    /**
     * Set the consuming flag of this interactor. This flag is a little
     * more complicated than in simple interactors: if not set, then
     * the event is consumed only if the clicked-on figure is added
     * to or removed from the selection. Otherwise it is not consumed.
     * If the flag is set, then the event is always consumed, thus
     * making it effectively "opaque" to events.
     *
     * <P> Note that the behaviour when the flag is false is the desired
     * behaviour when building panes that have an interactor attached
     * to the background. That way, the event passes through to the background
     * if a figure is hit on but the selection interactor's filters are
     * set up to ignore that particular event.
     *
     * <p> There is a third possibility, which is not supported: never
     * consume events. There is no way to do this currently, as the other
     * two behaviors seemed more likely to be useful. (Also, that behaviour
     * is harder to implement because of interaction with the superclass.)
     */
    public void setConsuming (boolean flag) {
        // This method is only here for documentation purposes
        super.setConsuming(flag);
    }

    public void setPanFilter(MouseFilter f) {
        _panFilter = f;
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
    
 
}



