/*
 * @(#)AbstractWizard.java
 *
 * Copyright (c) 2004 Wolfram Research Inc., All Rights Reserved.
 */
package com.wolfram.guikit.wizard;

import java.awt.Container;
import java.awt.Paint;
import java.util.Collection;
import java.util.Iterator;
import java.util.Vector;

import javax.swing.JDialog;
import javax.swing.event.EventListenerList;

import com.wolfram.guikit.wizard.event.WizardEvent;
import com.wolfram.guikit.wizard.event.WizardListener;
import com.wolfram.guikit.wizard.page.WizardPage;
import com.wolfram.guikit.wizard.ui.PageNavigationPanel;
import com.wolfram.guikit.wizard.ui.PageSideBarPanel;

/**
 * AbstractWizard
 *
 * @version $Revision: 1.3 $
 */
public abstract class AbstractWizard implements Wizard {
  
  private String title;

  private WizardContainer container;
  
  // This could be set here and shared across pages
  private PageSideBarPanel sideBarPanel;
  private Object sideBarImage;
  private Paint sideBarPaint;
  private Container sideBarContent;
  private String sideBarTitle;
  private JDialog helpDialog;
  
  // This could be set here and shared across pages
  private PageNavigationPanel navigationPanel;
  
  private Vector pages = new Vector();
  
  private WizardPage lastPage;
  
  protected boolean resultModeCalled = false;
  protected boolean resultCloseCalled = false;
  
  protected EventListenerList listeners = null;
  
  public void reset() {
		resultModeCalled = false;
		resultCloseCalled = false;
    fireWizardDidReset();
  	}
  
  public JDialog getHelpDialog() {return helpDialog;}
  public void setHelpDialog(JDialog dialog) {
    helpDialog = dialog;
    }
  
  public WizardPage getNextPage(WizardPage page) {
    if (page == null) return null;
    // Do we always allow a page to override Wizard page order?
    if (page.getNextPage() != null) return page.getNextPage();
    int loc = pages.indexOf(page);
    if (loc == -1 || loc == pages.size()-1) return null;
    return (WizardPage)pages.elementAt(loc + 1);
    }
  public WizardPage getPreviousPage(WizardPage page) {
    if (page == null) return null;
    // Do we always allow a page to override Wizard page order?
    if (page.getPreviousPage() != null) return page.getPreviousPage();
    int loc = pages.indexOf(page);
    if (loc <= 0) return null;
    return (WizardPage)pages.elementAt(loc - 1);
    }
  
  public WizardPage getLastPage() {
    if (lastPage != null) return lastPage;
    if (pages.size() == 0) return null;
    else return (WizardPage)pages.lastElement();
    }
  public void setLastPage(WizardPage p) {
    lastPage = p;
    }
    
  public WizardPage getPage(int index) {
    if (index >=0 && index < pages.size()) return (WizardPage)pages.elementAt(index);
    else return null;
    }
    
  public WizardPage[] getPages() {
    return (WizardPage[])pages.toArray(new WizardPage[]{});
    }

  public void setPages(WizardPage[] pges) {
    this.pages.clear();
    if (pges != null) {
      for (int i = 0; i < pges.length; ++i) {
        WizardPage thisPage = pges[i];
        if (thisPage == null) continue;
        addPage(thisPage);
        }
      }
    }
  
  public void addPages(Collection c) {
    Iterator it = c.iterator();
    while (it.hasNext()) {
      Object thisPage = it.next();
      if (thisPage == null || !(thisPage instanceof WizardPage)) continue;
      addPage((WizardPage)thisPage);
      }
    }
 
  public void addPage(WizardPage p) {
    pages.add(p);
    // Is this also where we setup the default previous and next links?
    p.setWizard(this);
    }
    
	/**
	 * @return
	 */
	public String getTitle() {
		return title;
	}

	/**
	 * @param string
	 */
	public void setTitle(String string) {
		title = string;
	}

  public WizardContainer getContainer() {return container;}
  public void setContainer(WizardContainer wizardContainer) {
    this.container = wizardContainer;
    resultModeCalled = false;
    resultCloseCalled = false;
    if (container != null && container.getWizard() != this) {
      container.setWizard(this);
      }
    }
  public WizardPage getCurrentPage(){
    if (container != null && container.getWizard() == this) return container.getCurrentPage();
    return null;
    }
  
  public PageSideBarPanel getSideBarPanel() {
    if (sideBarPanel != null) return sideBarPanel;
    if (container != null && container.getWizard() == this) return container.getSideBarPanel();
    return null;
    }
  public void setSideBarPanel(PageSideBarPanel panel) {
    sideBarPanel = panel;
    }
  
  public Object getSideBarImage() {return sideBarImage;}
  public void setSideBarImage(Object image) {
    sideBarImage = image;
    }
  
  public Paint getSideBarPaint() {
    return sideBarPaint;
    }
  public void setSideBarPaint(Paint p) {
    sideBarPaint = p;
    }
    
  public Container getSideBarContent() {return sideBarContent;}
  public void setSideBarContent(Container c) {
    sideBarContent = c;
    }
  
  public String getSideBarTitle() {return sideBarTitle;}
  public void setSideBarTitle(String t) {
    sideBarTitle = t;
    }
    
  public PageNavigationPanel getNavigationPanel() {
    if (navigationPanel != null) return navigationPanel;
    if (container != null && container.getWizard() == this) return container.getNavigationPanel();
    return null;
    }
  public void setNavigationPanel(PageNavigationPanel panel) {
    navigationPanel = panel;
    }
  
  public void setAllowBack(WizardPage page, boolean val) {
    if (getContainer() != null && getContainer().getCurrentPage() == page) {
      getContainer().updateNavigationActions();
      }
    }
  public void setAllowNext(WizardPage page, boolean val) {
    if (container != null && container.getCurrentPage() == page) {
      container.updateNavigationActions();
      }
    }
  public void setAllowLast(WizardPage page, boolean val) {
    if (container != null && container.getCurrentPage() == page) {
      container.updateNavigationActions();
      }
    }
  public void setAllowFinish(WizardPage page, boolean val) {
    if (container != null && container.getCurrentPage() == page) {
      container.updateNavigationActions();
      }
    }
  public void setAllowCancel(WizardPage page, boolean val) {
    if (container != null && container.getCurrentPage() == page) {
      container.updateNavigationActions();
      }
    }
  
  /**
   * Adds the specified WizardListener to receive WizardEvents.
   * <p>
   * Use this method to register a WizardListener object to receive
   * notifications when wizard changes occur
   *
   * @param l the WizardListener to register
   * @see #removeWizardListener(WizardListener)
   */
  public void addWizardListener(WizardListener l) {
    if (listeners == null) listeners = new EventListenerList();
    if ( l != null ) {
      listeners.add( WizardListener.class, l );
      }
    }

  /**
   * Removes the specified WizardListener object so that it no longer receives
   * WizardEvents.
   *
   * @param l the WizardListener to register
   * @see #addWizardListener(WizardListener)
   */
  public void removeWizardListener(WizardListener l) {
    if (listeners == null) return;
    if ( l != null ) {
      listeners.remove( WizardListener.class, l );
      }
    }
    
  public void fireWizardFinished() {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    WizardEvent e = null;
    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == WizardListener.class ) {
        if (e == null)
          e = new WizardEvent(this);
        ((WizardListener)lsns[i+1]).wizardFinished(e);
        }
      }
    }
    
  public void fireWizardClosed() {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    WizardEvent e = null;
    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == WizardListener.class ) {
        if (e == null)
          e = new WizardEvent(this);
        ((WizardListener)lsns[i+1]).wizardClosed(e);
        }
      }
    }
    
  public void fireWizardCanceled() {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    WizardEvent e = null;
    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == WizardListener.class ) {
        if (e == null)
          e = new WizardEvent(this);
        ((WizardListener)lsns[i+1]).wizardCanceled(e);
        }
      }
    }
    
  public void fireWizardDidReset() {
    if (listeners == null) return;
    Object[] lsns = listeners.getListenerList();
    WizardEvent e = null;
    // Process the listeners last to first, notifying
    // those that are interested in this event
    for ( int i = lsns.length - 2; i >= 0; i -= 2 ) {
      if ( lsns[i] == WizardListener.class ) {
        if (e == null)
          e = new WizardEvent(this);
        ((WizardListener)lsns[i+1]).wizardDidReset(e);
        }
      }
    }
    
  // This would involve telling pages we are finishing and
  // producing the ending results of a finish event
  // Definitions would have to still use an endModal for assigning
  // return results if a valid finish is called
  public void finish() {
    if (!resultModeCalled) {
      resultModeCalled = true;
      fireWizardFinished();
      }
    }
  
  // Closing does not include a preceeding finish or cancel call unless
  // one gets a close wihout either then it is assumed this is a cancel
  public void close() {
    if (!resultModeCalled) cancel();
    if (!resultCloseCalled) {
      resultCloseCalled = true;
      fireWizardClosed();
      }
    }
  
  // This may do nothing or at least allow listeners to make sure no
  // successful results are returned
  public void cancel() {
    if (!resultModeCalled) {
      resultModeCalled = true;
      fireWizardCanceled();
      }
    }
    
}