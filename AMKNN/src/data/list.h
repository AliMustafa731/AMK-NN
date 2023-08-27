#pragma once

#include <cassert>
#include "common.h"

//----------------------------------
//   Double Linked-List
//----------------------------------
template<typename T> struct List
{
    struct Node
    {
        T value;
        Node *next, *prev;

        Node()
        {
            next = NULL;
            prev = NULL;
        }
        Node(T val)
        {
            value = val;
            next = NULL;
            prev = NULL;
        }
    };
    
    static inline bool compareNode(Node *a, Node *b)
    {
        return (a->next == b->next && a->prev == b->prev);
    }

    Node *base, *head;
    int length;

    List() { length = 0; }

    void add(T val)  // add node at the end of the list
    {
        if (length < 1)  // the first addition
        {
            base = new Node(val);
            head = base;
        }
        else
        {
            head->next = new Node(val);
            head->next->prev = head;
            head = head->next;
        }

        length += 1;
    }

    void remove(Node *node)
    {
        bool external_node = true;

        for (Node *n = base; n->next != NULL; n = n->next)
        {
            // check if the node is within our linked list
            if (compareNode(n, node))
            {
                external_node = false;
                break;
            }
        }
        if (external_node || node == NULL) { return; }


        if (node->prev == NULL)  // the first node
        {
            Node *temp = base;

            if (base->next != NULL)
            {
                base->next->prev = NULL;
                base = base->next;
            }
            else
            {
                base = NULL;
            }

            delete temp;
        }
        else if (node->next == NULL)  // the last node
        {
            Node *temp = head;

            head->prev->next = NULL;
            head = head->prev;

            delete temp;
        }
        else  // intermediate node
        {
            node->next->prev = node->prev;
            node->prev->next = node->next;
            delete node;
        }

        length -= 1;
    }

    void release()
    {
        Node *n = base;

        while (n != NULL)
        {
            Node *temp = n;
            n = n->next;
            delete temp;
        }

        length = 0;
    }
};
