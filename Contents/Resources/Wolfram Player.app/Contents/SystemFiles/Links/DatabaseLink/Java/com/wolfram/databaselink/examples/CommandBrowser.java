package com.wolfram.databaselink.examples;

import com.odellengineeringltd.glazedlists.*;
import com.odellengineeringltd.glazedlists.jtable.*;
import com.odellengineeringltd.glazedlists.query.*;

import java.awt.BorderLayout;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.Statement;

import javax.swing.*;

public class CommandBrowser extends JFrame
{
  protected String path;

  protected DynamicQueryList commands;
  protected CaseInsensitiveFilterList filteredCommands;
  protected SortedList sortedList;
  protected ListTable listTable;

  protected CommandIdComparator sortById = new CommandIdComparator();
  protected CommandFullFormComparator sortByFullForm = new CommandFullFormComparator();
  protected CommandDateTimeComparator sortByDateTime = new CommandDateTimeComparator();

  protected JPopupMenu popup;
  protected JToolBar toolBar;
  protected JMenuItem pasteMenuItem;
  protected JLabel statusLabel = new JLabel("      ");

  protected PopupListener popupListener;
  protected Action pasteAction;
  protected Action deleteAction;
  protected Action clearAction;

  public CommandBrowser() throws Exception
  {
    new CommandBrowser(null);
  }

  public CommandBrowser(String databasePath) throws Exception
  {

    path = databasePath;

    commands = new DynamicQueryList(1000);
    commands.setQuery(new CommandQuery(databasePath));

    sortedList = new SortedList(commands, sortById);
    filteredCommands = new CaseInsensitiveFilterList(sortedList);
    listTable = new ListTable(filteredCommands, new CommandTableCell());

    JLabel filterText = new JLabel("Filter: ");

    toolBar = new JToolBar();
    toolBar.setLayout(new BorderLayout());
    toolBar.add(filterText, BorderLayout.WEST);
    toolBar.add(filteredCommands.getFilterEdit(), BorderLayout.CENTER);

    // add to the container
    getContentPane().setLayout(new BorderLayout());
    getContentPane().add(toolBar, BorderLayout.NORTH);
    getContentPane().add(listTable.getTableScrollPane(), BorderLayout.CENTER);
    getContentPane().add(statusLabel, BorderLayout.SOUTH);

    setSize(640, 480);

    popup = new JPopupMenu();

    pasteAction = new AbstractAction()
    {
      public void actionPerformed(ActionEvent evt)
      {
        //paste code here
      }
    };

    deleteAction = new AbstractAction()
    {
      public void actionPerformed(ActionEvent evt)
      {
        Connection conn = null;
        try
        {
          Class.forName("org.hsqldb.jdbcDriver");
          conn = DriverManager.getConnection("jdbc:hsqldb:" + path, "sa", "");

          PreparedStatement stmt = conn.prepareStatement("DELETE FROM COMMANDS WHERE ID = ?");

          EventList selections = listTable.getSelectionList();
          int numrows = selections.size();
          for (int i = 0; i < numrows; i++)
          {
            stmt.setInt(1, ((Command)selections.get(i)).getId());
            stmt.executeUpdate();
          }
          conn.close();
        }
        catch(Exception e)
        {
          e.printStackTrace();
          try
          {
            if(conn != null)
              conn.close();
          } catch(Exception f) {}
        }
      }
    };

    clearAction = new AbstractAction()
    {
      public void actionPerformed(ActionEvent evt)
      {
        Connection conn = null;
        try
        {
          Class.forName("org.hsqldb.jdbcDriver");
          conn = DriverManager.getConnection("jdbc:hsqldb:" + path, "sa", "");

          Statement stmt = conn.createStatement();
          stmt.execute("DROP TABLE COMMANDS");
          conn.close();
        }
        catch(Exception e)
        {
          e.printStackTrace();
          try
          {
            if(conn != null)
              conn.close();
          } catch(Exception f) {}
        }
      }
    };

    pasteMenuItem = new JMenuItem();
    pasteMenuItem.setText("Paste");
    popup.add(pasteMenuItem);

    JMenuItem menuItem = new JMenuItem();
    menuItem.setAction(deleteAction);
    menuItem.setText("Delete");
    popup.add(menuItem);

    menuItem = new JMenuItem();
    menuItem.setAction(clearAction);
    menuItem.setText("Clear");
    popup.add(menuItem);

    JMenu submenu = new JMenu("Sort");
    popup.add(submenu);

    ButtonGroup methodGroup = new ButtonGroup();
    menuItem = new JCheckBoxMenuItem("Id", true);
    menuItem.addActionListener(new ActionListener()
    {
      public void actionPerformed(ActionEvent e)
      {
        sortedList.setComparator(sortById);
      }
    });
    methodGroup.add(menuItem);
    submenu.add(menuItem);

    menuItem = new JCheckBoxMenuItem("FullForm", false);
    menuItem.addActionListener(new ActionListener()
    {
      public void actionPerformed(ActionEvent e)
      {
        sortedList.setComparator(sortByFullForm);
      }
    });
    methodGroup.add(menuItem);
    submenu.add(menuItem);

    menuItem = new JCheckBoxMenuItem("DateTime", false);
    menuItem.addActionListener(new ActionListener()
    {
      public void actionPerformed(ActionEvent e)
      {
        sortedList.setComparator(sortByDateTime);
      }
    });
    methodGroup.add(menuItem);
    submenu.add(menuItem);

    popupListener = new PopupListener();
    listTable.getTableScrollPane().addMouseListener(popupListener);
    listTable.getTable().addMouseListener(popupListener);

  }

  public Command[] getSelectedItems()
  {
    EventList list = listTable. getSelectionList();
    return (Command[])list.toArray(new Command[list.size()]);
  }

  public Command[] getItems()
  {
    return (Command[])commands.toArray(new Command[commands.size()]);
  }

  public JMenuItem getPasteMenuItem()
  {
    return pasteMenuItem;
  }

  private class PopupListener extends MouseAdapter
  {
    public void mousePressed(MouseEvent e)
    {
      maybeShowPopup(e);
    }
    public void mouseReleased(MouseEvent e)
    {
      maybeShowPopup(e);
    }

    private void maybeShowPopup(MouseEvent e)
    {
      if (e.isPopupTrigger())
      {
        EventList selections = listTable.getSelectionList();
        int numrows = selections.size();

        if (pasteAction != null) pasteAction.setEnabled(numrows == 1 ? true : false);
        if (deleteAction != null) deleteAction.setEnabled(numrows > 0 ? true : false);
        if (clearAction != null) clearAction.setEnabled(true);

        popup.show(e.getComponent(), e.getX(), e.getY());
      }
    }
  }

}
