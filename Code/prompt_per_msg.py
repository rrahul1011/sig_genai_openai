customer_style = """British English \
            thrilled and delighted
            """

instruction_existing= """
                1. Create a captivating message within an 80-word limit for maximum impact.
                2. Add relevant emojis and apply formatting to enhance the message's appeal.
                3. Maintain clear structure with suitable line breaks.
                5. Focus solely on the message content; avoid extra information.
                6. Ensure the message grabs attention.
                7. Limit consecutive lines to a maximum of two for readability.
                8. Format the message crisply for an attention-grabbing effect.
                9. Use appropriate spacing between lines to increase attractiveness.
                10. Avoid more than two consecutive sentences in the message toghter use proper space in between sentences
                11. Use proper line breaker.
                12. Give the offer and recommendation in point form in attractive manner
                """

template_string = """Generate a personalized welcome message for users logging into the 'Diageo' website.
                    Utilize existing user data, including {Existing_user_data}, 
                    to provide tailored product recommendations ({Rec_product}). 
                    Incorporate cart item promotions and offers ({Offers_and_promotion}). 
                    Present the message in a stylish format ({style}). 
                    Follow the provided instructions carefully: {instruction_existing}.
                """



template_string_new ="""Generate a personalized welcome message for new users as they log in to the 'Diageo' website. 
                        Utilize user data ({user_data}) to tailor the message. 
                        Highlight the best-selling products ({best_selling_product}) 
                        and current welcome offers ({welcome_offer}). 
                        Present the message in a stylish format that is {style}
                        Please adhere to the provided instructions: {instruction_existing}.
                        """




best_selling_product= """
    The best selling products:-
    1.Ciroc Vodka
    2.Black & White Blended Whisky
    """


welcome_offer="""1.Free express shipping for a limited time.
                        2.Give $10, get $10 when you refer a friend."""